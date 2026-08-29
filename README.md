# Maple on mlx-lm

Maple is a 20B-A1B ternary MoE with 24 layers, 256 experts, top-8, 512-token sliding
window on 3 of every 4 layers. Weights are 2-bit packed `{-α, 0, +α}`, one α per
row. 

This fork runs on the stock MLX build for portability. We intend to release a faster custom
library in the coming days.

## Setup

Requires Apple Silicon and [uv](https://docs.astral.sh/uv/).

```sh
git clone git@github.com:eli32-vlc/maple-flash-moe.git
cd maple-flash-moe
./setup.sh
source .venv/bin/activate
hf download deepgrove/maple-2bit-mlx --local-dir maple-2bit-mlx
```

## Run

```sh
python -m mlx_lm generate --model ./maple-2bit-mlx --trust-remote-code --flash-head \
  --prompt "Write a haiku about a grove." --temp 1.0 --top-p 0.95 --top-k 20

python -m mlx_lm chat --model ./maple-2bit-mlx --trust-remote-code --max-tokens -1 \
  --temp 1.0 --top-p 0.95
```

Enable flash head for extra speed.
```sh
python -m mlx_lm chat --model ./maple-2bit-mlx --trust-remote-code --max-tokens -1 \
  --temp 1.0 --top-p 0.95 --flash-head
``` 

| chip | head | decode tok/s | prefill tok/s | peak |
| --- | --- | --- | --- | --- |
| M4 | exact (default) | 169 | 1075 | 6.51 GB |
| M4 | `--flash-head` | **218** | 1075 | 6.69 GB |
| M5 Pro | exact (default) | 359 | 3773 | 6.73 GB |
| M5 Pro | `--flash-head` | **395** | 3857 | 6.92 GB |

## Convert

```sh
python -m mlx_lm.ternary /path/to/maple-bf16 -o maple-2bit-mlx --flash-head
```

Streams and converts shard by shard, so the 38 GB bf16 source is never fully resident.

- `--flash-head` — ~2 min of k-means, score 4748
  vocabulary-cluster centroids, then compute exact logits only for the top 512
  clusters (special tokens always scored). Greedy is exact whenever the true
  argmax is in a probed cluster. Attach to an already-converted
  directory with `python -m mlx_lm.ternary maple-2bit-mlx --flash-head-only`
  (rewrites in place; point it at a real directory, not hardlinks).
- `--group-scales` — repeat each row's α across every group (+0.6 GB), only for
  tools that read MLX quantized checkpoints generically. Default stores the row
  scale once as `row_alpha`; `sanitize()` expands it at load.

## Flash-MoE expert offload

Maple's 256 experts × 24 layers are the bulk of the resident memory: the full
model sits at **~5.9–6.5 GB** on Apple Silicon. Flash-MoE keeps only a small
*active* set of experts in DRAM and streams the rest from disk on demand.

How it works:

- **File-backed checkpoint.** Expert weights stay on disk. A byte-range reader
  (`_DiskHolder` in `mlx_lm/models/switch_layers.py`) seeks to and reads only the
  rows for the currently-active experts via numpy — the full 256-expert tensor is
  never materialized.
- **IFP (Inactive-Expert-Free Policy).** The gating function is masked to the
  active set, so only resident experts are ever scored.
- **Per-token reselect.** The active set is recomputed from the *current token's*
  true top-k routing (default every token via `--reselect-every 1`). A small LRU
  buffer in `_flash_page` keeps recently-used experts warm across tokens.
- **Fused projections.** The checkpoint stores unfused `up_proj`/`gate_proj` plus
  a per-row `row_alpha`; the reader concatenates the two source rows per active
  expert and expands `row_alpha` to per-group scales/biases (BF16, bit-reinterpreted).

### Lossless at the default

With `--active-experts 8 --shared-experts 0` and per-token decode, the active set
*is* the model's true top-8 routing, so generated tokens are **bit-exact** vs the
full model (verified: weight/scales/biases max diff = 0.0). Larger
`--active-experts` trades a little quality for headroom; `--active-experts 256`
reproduces the full model exactly.

> Prefill note: during batch prefill every token routes to its own top-k set, so
> the union of experts can far exceed `active`. The prefill pass keeps that full
> union resident (only decode caps to `--active-experts`), which is what makes
> prefill exact; resident memory at generation time is still bounded by the
> activated set.

| config | peak RAM | notes |
| --- | --- | --- |
| full (no flash) | ~5.9–6.5 GB | baseline |
| `--flash-moe --active-experts 8 --shared-experts 0` | **0.63 GB** | lossless (top-8 routing) |
| `--flash-moe --active-experts 32 --shared-experts 2` | 0.94 GB | coherent, slight quality cost |
| `--flash-moe --active-experts 256` | 5.89 GB | equals full model |

> All flash-MoE memory numbers measured on a MacBook Air (Apple M2, 16 GB
> unified memory, stock MLX build).

> Note on `--shared-experts`: quality loss is zero whenever
> `active >= top_k + shared`. Since Maple's `top_k = 8`, keep `--shared-experts 0`
> at the default `active = 8`, or raise `--active-experts` above 8 if you enable
> shared experts.

### Run with flash-MoE

```sh
python -m mlx_lm generate --model ./maple-2bit-mlx --trust-remote-code \
  --flash-moe --active-experts 8 --shared-experts 0 \
  --prompt "Write a haiku about a grove." --temp 0.7 --top-p 0.9 --top-k 40

python -m mlx_lm chat --model ./maple-2bit-mlx --trust-remote-code \
  --flash-moe --active-experts 8 --kv-bits 8 --kv-v-bits 4 --max-tokens -1
```

## KV cache quantization

The KV cache can be quantized independently for keys and values:

- `--kv-bits 8 --kv-v-bits 4` → 8-bit keys, 4-bit values (recommended default).
  Wired through `mlx_lm/models/cache.py` and `mlx_lm/models/base.py` with separate
  `k_bits`/`v_bits`.
- Sliding-window layers use `RotatingKVCache`, which does not support KV
  quantization yet, so those layers keep full-precision KV.

## Server (OpenAI-compatible)

The server exposes OpenAI-style `/v1/chat/completions` and `/v1/completions` and
supports every flash-MoE / KV-quant flag:

```sh
python -m mlx_lm server --model ./maple-2bit-mlx --trust-remote-code \
  --flash-moe --active-experts 8 --shared-experts 0 \
  --kv-bits 8 --kv-v-bits 4 \
  --port 8080 --prefill-step-size 8192 --max-tokens 8192
```

It logs live progress: prefill percentage + tok/s, and per-phase
`Prompt: N tokens, X tok/s` / `Generation: N tokens, Y tok/s`.

> Sampling matters for 2-bit weights: the default server `temp=0` (greedy) can
> loop or degenerate on a 2-bit model. Use `--temp 0.7 --top-p 0.9 --top-k 40
> --min-p 0.05` for coherent output.

## Diff vs upstream mlx-lm

| file | what |
| --- | --- |
| `mlx_lm/models/maple.py` | the model (also copied into every converted checkpoint) |
| `mlx_lm/ternary.py` | bf16 → ternary converter + FlashHead generator |
| `tests/test_maple_kernels.py` | kernel + precision self-check — `pytest tests/test_maple_kernels.py -v` |
| `generate.py`, `chat.py`, `server.py`, `benchmark.py` | support for `--flash-head` flag |
| `mlx_lm/models/switch_layers.py` | flash-MoE disk holder + paging — byte-range expert streaming, IFP, per-token reselect |
| `mlx_lm/models/cache.py`, `mlx_lm/models/base.py` | q8/q4 KV cache quantization (separate `k_bits`/`v_bits`) |
| `mlx_lm/models/maple.py` | `FlashMoE`, `_select` (per-token top-k active set), `prepare_flash_moe` |
| `generate.py`, `server.py` | `--flash-moe`, `--active-experts`, `--shared-experts`, `--reselect-every`, `--kv-bits`, `--kv-v-bits` flags + server tok/s metrics |
| `setup.sh` | uv venv + editable install |
