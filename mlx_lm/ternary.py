# Copyright © 2026 DeepGrove AI.
"""Convert a bf16 Maple checkpoint to 2-bit ternary MLX format.

Maple is trained quantization-aware, so its bf16 weights are recovered to
ternary values {-alpha_row, 0, +alpha_row} by thresholding (this is not
round-to-nearest quantization; `mlx_lm.convert` cannot produce it). The output
is a standard mlx-lm quantized checkpoint: packed 2-bit affine weights with
codes {0, 1, 2} and per-row scales alpha / biases -alpha, loadable by stock
mlx-lm with no custom kernels.

The source may be a local directory or a Hugging Face repo id. Repo inputs are
streamed shard by shard so the full bf16 model is never resident on disk (each
shard is downloaded, converted, then deleted).

Usage:
    python -m mlx_lm.ternary deepgrove/Maple-20B-A1B -o /path/to/maple-2bit-mlx
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import mlx.core as mx

DEFAULT_THRESHOLD_SCALE = 0.7
GROUP_SIZE = 128
HEAD_GROUP_SIZE = 64
HEAD_BITS = 4
SHARD_BYTES = 2 << 30
SAFETENSORS_METADATA = {"format": "mlx"}

# Not ternarized: the router runs in float32. (The head and embeddings are
# claimed by RTN_KEYS before the ternary check runs; norms are 1D.)
TERNARY_EXCLUDE = (".mlp.gate.weight",)
RTN_KEYS = ("lm_head.weight", "model.word_embeddings.weight")
# Internal training artifacts that must not ship.
SKIP_AUX = (
    "modeling_maple.py",
    "configuration_maple.py",
    "fa3.py",
    "fa3_utils.py",
    "model.safetensors.index.json",
)

_SHIFTS = mx.arange(0, 32, 2, dtype=mx.uint32)


def _pack_2bit(codes: mx.array) -> mx.array:
    """Pack 2-bit codes along the last axis, 16 per uint32, LSB first.

    This matches the packing `mx.quantize(..., bits=2, mode="affine")`
    produces, verified bit-exact against `mx.dequantize`.
    """
    *lead, k = codes.shape
    codes = codes.astype(mx.uint32).reshape(*lead, k // 16, 16)
    return mx.sum(codes << _SHIFTS, axis=-1).astype(mx.uint32)


def ternarize(weight: mx.array, threshold_scale: float = DEFAULT_THRESHOLD_SCALE):
    """Ternarize [..., N, K] weights row-wise; return (packed, scales, biases).

    threshold = threshold_scale * mean(|w|) per row; surviving weights carry
    their sign and the row scale alpha = mean(|w| surviving).

    The arithmetic runs in float32 regardless of the weight dtype, matching
    the quantizer the model was trained with. Doing it in bf16 instead moves
    the threshold enough to flip ~0.15% of the codes and perturbs alpha by up
    to ~4%, since the reduction over K rounds every partial sum. Only the
    final alpha is rounded back to the weight dtype.
    """
    if weight.shape[-1] % GROUP_SIZE:
        raise ValueError(
            f"Ternary conversion requires the reduction dim to be a multiple "
            f"of {GROUP_SIZE}; got {weight.shape[-1]}."
        )
    w = weight.astype(mx.float32)
    aw = mx.abs(w)
    threshold = threshold_scale * mx.mean(aw, axis=-1, keepdims=True)
    mask = (aw > threshold).astype(mx.float32)
    alpha_num = mx.sum(aw * mask, axis=-1, keepdims=True)
    alpha_den = mx.maximum(mx.sum(mask, axis=-1, keepdims=True), 1)
    alpha = (alpha_num / alpha_den).astype(weight.dtype)

    ternary = mx.sign(w) * mask
    packed = _pack_2bit(ternary + 1)

    k = weight.shape[-1]
    scales = mx.contiguous(
        mx.broadcast_to(alpha, (*weight.shape[:-1], k // GROUP_SIZE))
    )
    return packed, scales, -scales


class _ShardWriter:
    """Accumulates tensors and flushes ~2 GB safetensors shards."""

    def __init__(self, output: Path):
        self.output = output
        self.tensors = {}
        self.nbytes = 0
        self.weight_map = {}
        self.shard_names = []
        self.total_size = 0

    def add(self, key: str, value: mx.array):
        self.tensors[key] = value
        self.nbytes += value.nbytes
        if self.nbytes >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.tensors:
            return
        mx.eval(*self.tensors.values())
        name = f"model-part-{len(self.shard_names):05d}.safetensors"
        path = self.output / name
        mx.save_safetensors(str(path), self.tensors, metadata=SAFETENSORS_METADATA)
        for k in self.tensors:
            self.weight_map[k] = name
        self.shard_names.append(name)
        self.total_size += path.stat().st_size
        self.tensors = {}
        self.nbytes = 0
        gc.collect()

    def finalize(self):
        self.flush()
        n = len(self.shard_names)
        if n == 1:
            final = {self.shard_names[0]: "model.safetensors"}
        else:
            final = {
                name: f"model-{i + 1:05d}-of-{n:05d}.safetensors"
                for i, name in enumerate(self.shard_names)
            }
        for old, new in final.items():
            (self.output / old).rename(self.output / new)
        weight_map = {k: final[v] for k, v in self.weight_map.items()}
        index = {"metadata": {"total_size": self.total_size}, "weight_map": weight_map}
        (self.output / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2)
        )


def _is_ternary_target(key: str, shape) -> bool:
    return (
        key.endswith(".weight")
        and len(shape) == 2
        and not any(pat in key for pat in TERNARY_EXCLUDE)
    )


class _Converter:
    def __init__(
        self,
        output: Path,
        config: dict,
        threshold_scale: float,
        group_scales: bool = False,
    ):
        self.writer = _ShardWriter(output)
        self.threshold_scale = threshold_scale
        self.group_scales = group_scales
        self.num_experts = config["num_experts"]
        self.tie_word_embeddings = config.get("tie_word_embeddings", False)
        # model.layers.{l}.mlp.experts.{e}.{proj}.weight accumulate here until
        # a full expert set is present, then are stacked into switch_mlp form.
        self.pending = {}

    def _emit_quantized(self, prefix: str, packed, scales, biases):
        self.writer.add(f"{prefix}.weight", packed)
        self.writer.add(f"{prefix}.scales", scales)
        self.writer.add(f"{prefix}.biases", biases)

    def _emit_ternary(self, prefix: str, packed, scales, biases):
        """Ternary tensors carry one scale per output row and bias == -scale,
        so every group in a row repeats the same value. Store just that value
        as `row_alpha` (~0.6 GB smaller); maple.py expands it at load.
        `group_scales` writes the repeated per-group tensors instead, for
        tools that read MLX quantized checkpoints generically."""
        if self.group_scales:
            self._emit_quantized(prefix, packed, scales, biases)
            return
        self.writer.add(f"{prefix}.weight", packed)
        self.writer.add(f"{prefix}.row_alpha", mx.contiguous(scales[..., 0]))

    def _convert_expert(self, key: str, value: mx.array):
        # key: model.layers.{l}.mlp.experts.{e}.{proj}.{param}
        parts = key.split(".")
        layer, expert, proj, param = parts[2], int(parts[5]), parts[6], parts[7]
        slot = self.pending.setdefault((layer, proj, param), {})
        slot[expert] = value
        if len(slot) < self.num_experts:
            return
        stacked = mx.stack([slot[e] for e in range(self.num_experts)])
        del self.pending[(layer, proj, param)]
        prefix = f"model.layers.{layer}.mlp.switch_mlp.{proj}"
        if param == "weight":
            packed, scales, biases = ternarize(stacked, self.threshold_scale)
            mx.eval(packed, scales, biases)
            self._emit_ternary(prefix, packed, scales, biases)
        else:
            # e.g. per-expert biases: stack and pass through unquantized.
            mx.eval(stacked)
            self.writer.add(f"{prefix}.{param}", stacked)

    def add(self, key: str, value: mx.array):
        if key.startswith("lm_head.") and self.tie_word_embeddings:
            return  # tied models have no head; maple.py uses the embeddings
        if ".mlp.experts." in key:
            # Materialize now: the source shard is deleted after processing.
            mx.eval(value)
            self._convert_expert(key, value)
        elif key in RTN_KEYS:
            packed, scales, biases = mx.quantize(
                value, group_size=HEAD_GROUP_SIZE, bits=HEAD_BITS
            )
            mx.eval(packed, scales, biases)
            self._emit_quantized(key[: -len(".weight")], packed, scales, biases)
        elif _is_ternary_target(key, value.shape):
            packed, scales, biases = ternarize(value, self.threshold_scale)
            mx.eval(packed, scales, biases)
            self._emit_ternary(key[: -len(".weight")], packed, scales, biases)
        else:
            # Materialize before the source shard is deleted.
            mx.eval(value)
            self.writer.add(key, value)

    def finalize(self):
        if self.pending:
            missing = sorted(self.pending)
            raise RuntimeError(f"Incomplete expert sets at end of stream: {missing}")
        self.writer.finalize()


def _write_config(source_config: dict, output: Path):
    config = dict(source_config)
    config["model_type"] = "maple"
    config["model_file"] = "maple.py"
    # The training code routes with a plain softmax + top-k + renormalize, so
    # these inherited grouped-routing fields describe behaviour that does not
    # exist. ModelArgs ignores them; drop them rather than ship them as noise.
    for stale in ("score_function", "n_group", "topk_group", "routed_scaling_factor"):
        config.pop(stale, None)
    config.pop("auto_map", None)
    config.pop("quantize", None)
    config.pop("quantization_config", None)
    quantization = {
        "group_size": GROUP_SIZE,
        "bits": 2,
        "mode": "affine",
        "lm_head": {"group_size": HEAD_GROUP_SIZE, "bits": HEAD_BITS},
        "model.word_embeddings": {"group_size": HEAD_GROUP_SIZE, "bits": HEAD_BITS},
    }
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))


def _copy_model_file(output: Path):
    src = Path(__file__).parent / "models" / "maple.py"
    shutil.copy2(src, output / "maple.py")


def _shard_names(index_path: Path) -> list:
    weight_map = json.loads(index_path.read_text())["weight_map"]
    return sorted(set(weight_map.values()))


def _stage_local(source: Path, output: Path) -> list:
    index = source / "model.safetensors.index.json"
    if index.exists():
        shard_names = _shard_names(index)
    else:
        shard_names = sorted(p.name for p in source.glob("*.safetensors"))
        if not shard_names:
            raise SystemExit(f"No .safetensors found in {source}")
    for f in source.iterdir():
        if f.is_dir() or f.suffix == ".safetensors" or f.name in SKIP_AUX:
            continue
        shutil.copy2(f, output / f.name)
    return shard_names


def _stage_hf(repo_id: str, output: Path) -> list:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id)
    for fn in files:
        if (
            fn.endswith(".safetensors")
            or fn in SKIP_AUX
            or fn.startswith(("__pycache__", "."))
        ):
            continue
        hf_hub_download(repo_id=repo_id, filename=fn, local_dir=str(output))
    if "model.safetensors.index.json" in files:
        idx = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors.index.json",
            local_dir=str(output / "_src"),
        )
        return _shard_names(Path(idx))
    shard_names = sorted(f for f in files if f.endswith(".safetensors"))
    if not shard_names:
        raise SystemExit(f"No .safetensors found in {repo_id}")
    return shard_names


def convert(
    source: str,
    output: str,
    threshold_scale: float = DEFAULT_THRESHOLD_SCALE,
    keep_source: bool = False,
    group_scales: bool = False,
):
    output = Path(output)
    if output.exists() and list(output.glob("model*.safetensors")):
        raise SystemExit(
            f"Output directory {output} already contains model shards; "
            "remove them or choose a new directory (stale shards would be "
            "loaded together with the new ones)."
        )
    output.mkdir(parents=True, exist_ok=True)

    is_local = Path(source).is_dir()
    if is_local:
        source_dir = Path(source)
        shard_names = _stage_local(source_dir, output)
        get_shard = lambda name: source_dir / name  # noqa: E731
        drop_shard = lambda path: None  # noqa: E731
    else:
        from huggingface_hub import hf_hub_download

        shard_names = _stage_hf(source, output)
        src_dir = output / "_src"
        src_dir.mkdir(exist_ok=True)
        get_shard = lambda name: Path(  # noqa: E731
            hf_hub_download(repo_id=source, filename=name, local_dir=str(src_dir))
        )
        drop_shard = lambda path: None if keep_source else path.unlink()  # noqa: E731

    source_config = json.loads((output / "config.json").read_text())
    converter = _Converter(
        output, source_config, threshold_scale, group_scales=group_scales
    )

    for i, name in enumerate(shard_names, 1):
        print(f"[{i}/{len(shard_names)}] {name}", flush=True)
        path = get_shard(name)
        tensors = mx.load(str(path))
        for key in list(tensors.keys()):
            converter.add(key, tensors.pop(key))
        del tensors
        gc.collect()
        drop_shard(path)

    converter.finalize()
    _write_config(source_config, output)
    _copy_model_file(output)
    if not is_local:
        if keep_source:
            print(f"bf16 source shards kept in {output / '_src'}", flush=True)
        else:
            shutil.rmtree(output / "_src", ignore_errors=True)

    size = converter.writer.total_size
    print(f"Done: {size / 1e9:.2f} GB -> {output}", flush=True)


def _balanced_spherical_kmeans(W, n_clusters, n_iter=60, seed=42, chunk=16384):
    """Balanced spherical k-means over lm_head rows (equal-size clusters)."""
    import numpy as np

    N, D = W.shape
    cluster_size = N // n_clusters
    rng = np.random.RandomState(seed)

    X = mx.array(W)
    C = mx.array(W[rng.choice(N, n_clusters, replace=False)])
    C = C / (mx.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    mx.eval(X, C)

    best_obj, best_centroids, best_labels = -1.0, None, None
    for it in range(n_iter):
        # Nearest centroid (chunked on GPU).
        labels = np.zeros(N, dtype=np.int32)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            labels[s:e] = np.array(mx.argmax(X[s:e] @ C.T, axis=1))

        # Evict overflow from over-full clusters, keeping the closest members.
        counts = np.bincount(labels, minlength=n_clusters)
        over = np.where(counts > cluster_size)[0]
        if len(over):
            evict = []
            for ci in over:
                members = np.where(labels == ci)[0]
                sims = np.array(
                    (X[mx.array(members)] @ C[int(ci) : int(ci) + 1].T).squeeze(1)
                )
                keep = np.argpartition(sims, -cluster_size)[-cluster_size:]
                mask = np.ones(len(members), dtype=bool)
                mask[keep] = False
                evict.extend(members[mask].tolist())
                labels[members[mask]] = -1

            recount = np.bincount(labels[labels >= 0], minlength=n_clusters).astype(
                np.int32
            )
            full = recount >= cluster_size
            evict = np.array(evict, dtype=np.int64)
            for bs in range(0, len(evict), 2048):
                pts = evict[bs : bs + 2048]
                sims_b = np.array(X[mx.array(pts)] @ C.T)
                sims_b[:, full] = -2.0
                for i, pt in enumerate(pts):
                    bc = int(np.argmax(sims_b[i]))
                    if sims_b[i, bc] <= -2.0:
                        bc = int(np.argmax(recount < cluster_size))
                    labels[pt] = bc
                    recount[bc] += 1
                    if recount[bc] >= cluster_size:
                        full[bc] = True
                        sims_b[:, bc] = -2.0

        # Centroid update + objective. The scatter-add runs on the GPU:
        # np.add.at is unbuffered and costs ~16x more at this size, and X is
        # already resident.
        sums = mx.zeros((n_clusters, D), dtype=mx.float32).at[mx.array(labels)].add(X)
        counts_f = mx.array(
            np.maximum(np.bincount(labels, minlength=n_clusters), 1).astype(np.float32)
        )
        C = sums / counts_f[:, None]
        C = C / (mx.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
        mx.eval(C)

        obj = 0.0
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            obj += float(mx.sum(X[s:e] * C[mx.array(labels[s:e])]).item())
        obj /= N
        if obj > best_obj:
            best_obj = obj
            best_centroids = np.array(C.astype(mx.float32))
            best_labels = labels.copy()
        if (it + 1) % 5 == 0 or it < 3:
            print(f"  kmeans iter {it + 1}/{n_iter}: obj={obj:.6f}", flush=True)

    return best_centroids, best_labels


def generate_flash_head(
    model_dir: str,
    n_clusters: int = 4748,
    n_iter: int = 60,
    n_probes: int = 512,
    seed: int = 42,
    head_copy: bool = False,
):
    """Cluster the lm_head of a converted checkpoint and attach FlashHead data.

    Writes lm_head_flash.* tensors into a new shard and records the FlashHead
    metadata (including forced control tokens) in config.json.
    """
    import numpy as np

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("tie_word_embeddings"):
        raise SystemExit("FlashHead requires an untied lm_head.")
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    needed = ["lm_head.weight", "lm_head.scales", "lm_head.biases"]
    if any(k not in weight_map for k in needed):
        raise SystemExit(
            "Checkpoint has no quantized lm_head (lm_head.weight/scales/biases); "
            "cannot attach FlashHead."
        )
    tensors = {}
    for shard in sorted({weight_map[k] for k in needed}):
        data = mx.load(str(model_dir / shard))
        tensors.update({k: data[k] for k in needed if k in data})

    lm_q = config["quantization"].get("lm_head") or config["quantization"]
    W = mx.dequantize(
        tensors["lm_head.weight"],
        tensors["lm_head.scales"],
        tensors["lm_head.biases"],
        group_size=lm_q["group_size"],
        bits=lm_q["bits"],
    ).astype(mx.float32)
    W_np = np.array(W)
    row_norms = np.linalg.norm(W_np, axis=1, keepdims=True) + 1e-8
    # Cluster directions, not magnitudes: high-frequency tokens have small
    # rows but large cosines; without row normalization their clusters are
    # directionally incoherent and the probe phase misses them.
    W_np /= row_norms
    vocab_size = W_np.shape[0]
    if vocab_size % n_clusters:
        raise SystemExit(
            f"--clusters must divide the vocab size {vocab_size}; got {n_clusters}"
        )
    if n_probes > n_clusters:
        raise SystemExit(
            f"--probes ({n_probes}) must not exceed --clusters ({n_clusters})"
        )
    cluster_size = vocab_size // n_clusters

    print(
        f"FlashHead: clustering {vocab_size} x {W_np.shape[1]} lm_head into "
        f"{n_clusters} clusters of {cluster_size}",
        flush=True,
    )
    centroids, labels = _balanced_spherical_kmeans(W_np, n_clusters, n_iter, seed)

    token_map = np.zeros((n_clusters, cluster_size), dtype=np.int32)
    for c in range(n_clusters):
        members = list(np.where(labels == c)[0][:cluster_size])
        while len(members) < cluster_size:
            members.append(int(members[0]) if members else 0)
        token_map[c] = members

    # Probe scoring scale: the largest member row norm per cluster (see
    # FlashHead in models/maple.py), folded into the centroid rows so scoring
    # is a single matmul.
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
    cluster_scale = row_norms[:, 0][token_map].max(axis=1)
    cw, cscales, cbiases = mx.quantize(
        mx.array(centroids * cluster_scale[:, None]).astype(mx.bfloat16),
        group_size=HEAD_GROUP_SIZE,
        bits=HEAD_BITS,
    )

    # Control tokens whose exact logits are always computed so FlashHead can
    # never trap the model in an unterminated block or block end-of-sequence.
    force = []
    eos = config.get("eos_token_id")
    if eos is not None:
        force.extend(eos if isinstance(eos, list) else [eos])
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_dir)
        for tag in ("</think>", "<|im_end|>", "<|endoftext|>"):
            force.extend(tok.encode(tag, add_special_tokens=False))
    except Exception as e:  # tokenizer optional; EOS-only fallback
        print(f"  tokenizer unavailable ({e}); forcing EOS only", flush=True)
    force = [int(t) for t in dict.fromkeys(force) if 0 <= int(t) < vocab_size]

    # Cluster-ordered copy of the quantized head: FlashHead computes subset
    # logits with a single gather_qmm over the probed blocks (no per-step
    # row gathers). It is lm_head permuted by token_map and carries no new
    # information, so it is not written by default — maple.py rebuilds it at
    # load, saving ~175 MB of download and disk. `head_copy` restores it for
    # readers predating that support.
    head = {}
    if head_copy:
        order = mx.array(token_map.reshape(-1))
        head = {
            f"lm_head_flash.head.{k}": tensors[f"lm_head.{k}"][order].reshape(
                n_clusters, cluster_size, -1
            )
            for k in ("weight", "scales", "biases")
        }

    out = {
        "lm_head_flash.centroids.weight": cw,
        "lm_head_flash.centroids.scales": cscales,
        "lm_head_flash.centroids.biases": cbiases,
        "lm_head_flash.token_map": mx.array(token_map),
        **head,
    }
    mx.eval(*out.values())
    shard_name = "model-flashhead.safetensors"
    shard_path = model_dir / shard_name
    # Idempotent rerun: replace a previous FlashHead shard's size contribution.
    old_size = shard_path.stat().st_size if shard_path.exists() else 0
    mx.save_safetensors(str(shard_path), out, metadata=SAFETENSORS_METADATA)
    # The shard is rewritten wholesale, so drop entries a previous run mapped
    # into it (e.g. head.* when rerun without --flash-head-copy).
    for k in [k for k, v in weight_map.items() if v == shard_name and k not in out]:
        del weight_map[k]
    for k in out:
        weight_map[k] = shard_name
    index["metadata"]["total_size"] += shard_path.stat().st_size - old_size
    index_path.write_text(json.dumps(index, indent=2))

    config["flash_head"] = {
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "n_probes": n_probes,
        "group_size": HEAD_GROUP_SIZE,
        "bits": HEAD_BITS,
        "head_group_size": lm_q["group_size"],
        "head_bits": lm_q["bits"],
        "scaled_centroids": True,
        "force_tokens": force,
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    # Refresh the standalone model file: FlashHead needs the current maple.py.
    _copy_model_file(model_dir)
    print(f"FlashHead attached ({n_clusters} clusters, force={force})", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source", help="Local checkpoint directory or Hugging Face repo id"
    )
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument(
        "--threshold-scale",
        type=float,
        default=DEFAULT_THRESHOLD_SCALE,
        help=f"Ternarization threshold scale (default {DEFAULT_THRESHOLD_SCALE})",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="For Hugging Face inputs, keep downloaded bf16 shards",
    )
    parser.add_argument(
        "--flash-head",
        action="store_true",
        help="Generate FlashHead clusters after conversion",
    )
    parser.add_argument(
        "--flash-head-only",
        action="store_true",
        help="Treat source as an already-converted directory; only attach FlashHead",
    )
    parser.add_argument(
        "--group-scales",
        action="store_true",
        help="Repeat each row's ternary scale across every group instead of "
        "storing it once as row_alpha (+0.6 GB). Only needed for tools that "
        "read MLX quantized checkpoints generically; this maple.py does not "
        "care either way.",
    )
    parser.add_argument(
        "--flash-head-copy",
        action="store_true",
        help="Also write FlashHead's cluster-ordered lm_head copy (+175 MB). "
        "It is a row-permutation of lm_head by token_map, rebuilt at load by "
        "default; use this only for readers predating that support.",
    )
    parser.add_argument("--clusters", type=int, default=4748, help="FlashHead clusters")
    parser.add_argument("--probes", type=int, default=512, help="FlashHead probes")
    parser.add_argument(
        "--kmeans-iters", type=int, default=60, help="FlashHead k-means iterations"
    )
    args = parser.parse_args()

    if args.flash_head_only:
        generate_flash_head(
            args.source,
            args.clusters,
            args.kmeans_iters,
            args.probes,
            head_copy=args.flash_head_copy,
        )
        return
    if args.output is None:
        parser.error("--output is required")
    convert(
        args.source,
        args.output,
        threshold_scale=args.threshold_scale,
        keep_source=args.keep_source,
        group_scales=args.group_scales,
    )
    if args.flash_head:
        generate_flash_head(
            args.output,
            args.clusters,
            args.kmeans_iters,
            args.probes,
            head_copy=args.flash_head_copy,
        )


if __name__ == "__main__":
    main()
