# Copyright © 2023-2024 Apple Inc.

import math
import numpy as np
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu


_SAFETENSORS_DTYPE = {
    "U32": (np.uint32, mx.uint32),
    "U8": (np.uint8, mx.uint8),
    "I32": (np.int32, mx.int32),
    "BF16": (np.uint16, mx.bfloat16),
    "F16": (np.float16, mx.float16),
    "F32": (np.float32, mx.float32),
}


def _build_file_index(model_path):
    """Map each safetensors tensor name to its file location (byte offset, dtype,
    shape) so expert rows can be read individually without materializing the
    whole (num_experts, ...) tensor."""
    import glob
    import json
    import os
    import struct

    index = {}
    for fp in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with open(fp, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        data_start = 8 + n
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            dt = meta["dtype"]
            if dt not in _SAFETENSORS_DTYPE:
                continue
            off0, _ = meta["data_offsets"]
            index[name] = (fp, data_start + off0, dt, tuple(meta["shape"]))
    return index


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


# ---------------------------------------------------------------------------
# Flash-MoE resident expert cache (AFM 3 Core Advanced style).
#
# When enabled, the full fused expert tensors stay file-backed (mmap) in
# `self.disk` (an opaque holder MLX does NOT treat as parameters) and only a
# small activated subset (size `active`) is materialized into DRAM in
# `self.weight`/`self.scales`/`self.bias`. Two modes:
#   * IFP (default): the gate is masked so it only routes to activated experts;
#     `set_active` pages the whole activated set up front.
#   * LRU fallback: the gate routes freely; any requested expert not resident is
#     paged in from disk, evicting the least-recently-used resident slot.
# ---------------------------------------------------------------------------


class _DiskHolder:
    """Opaque container holding the on-disk location of each expert tensor.

    MLX materializes an entire lazy tensor on any access (even a single row),
    so we must never keep the (num_experts, ...) tensor as an MLX array. Instead
    we store its file location and read only the requested expert rows via raw
    byte-range reads, which keeps all un-selected experts on disk. A fused
    projection (e.g. up_gate_proj) is read from its two source projections
    (up_proj, gate_proj) and concatenated per row.
    """

    def __init__(self):
        self.index = None

    def read(self, sf_key, ids):
        fp, abs_off, dt_str, shape = self.index[sf_key]
        np_dt, mlx_dt = _SAFETENSORS_DTYPE[dt_str]
        row_elems = int(np.prod(shape[1:]))
        row_bytes = row_elems * np.dtype(np_dt).itemsize
        out = np.empty((len(ids), *shape[1:]), dtype=np_dt)
        with open(fp, "rb") as f:
            for i, e in enumerate(ids):
                f.seek(abs_off + int(e) * row_bytes)
                f.readinto(out[i].reshape(-1))
        if mlx_dt == mx.bfloat16:
            # Reinterpret the 16-bit patterns as bfloat16 (bf16 == top half of
            # float32) rather than casting the integer bit values.
            out = (out.astype(np.uint32) << 16).view(np.float32)
            return mx.array(out, dtype=mx.float32).astype(mx.bfloat16)
        return mx.array(out, dtype=mlx_dt)


def _flash_init_resident(self, E, file_index, key_prefix, group_size=128):
    self.disk = _DiskHolder()
    self.disk.index = file_index
    self._flash_group_size = group_size
    self.real_num_experts = 256
    base = key_prefix.rsplit(".", 1)[0]  # '...model.layers.N.mlp.switch_mlp'
    name = key_prefix.rsplit(".", 1)[-1]
    # After sanitize, up_gate_proj = concat(up_proj, gate_proj); down_proj is
    # unfused. The checkpoint stores the source projections, not the fused one.
    proj_parts = ["up_proj", "gate_proj"] if name == "up_gate_proj" else ["down_proj"]
    w_keys = [f"{base}.{p}.weight" for p in proj_parts if f"{base}.{p}.weight" in file_index]
    a_keys = [f"{base}.{p}.row_alpha" for p in proj_parts if f"{base}.{p}.row_alpha" in file_index]
    self._src_weight = w_keys
    self._src_alpha = a_keys
    self._concat = len(w_keys) > 1
    self._disk_names = ["weight", "scales", "biases"]
    if not w_keys:
        return
    _, _, dt_str, wshape = file_index[w_keys[0]]
    _, w_dt = _SAFETENSORS_DTYPE[dt_str]
    out_dim = sum(file_index[w][3][1] for w in w_keys)
    packed = wshape[2]
    self.real_num_experts = wshape[0]
    self.weight = mx.zeros((E, out_dim, packed), dtype=w_dt)
    ngroups = packed * 16 // group_size
    self.scales = mx.zeros((E, out_dim, ngroups), dtype=mx.bfloat16)
    self.biases = mx.zeros((E, out_dim, ngroups), dtype=mx.bfloat16)
    self.active_ids = None
    self.slot_of = None
    self.cache_enabled = True
    self._age = mx.zeros((E,), dtype=mx.uint32)
    self._tick = 0


def _flash_set_active(self, ids):
    ids = mx.array(ids, dtype=mx.int32)
    self.active_ids = ids
    idx = mx.arange(self.real_num_experts)
    matches = (idx[:, None] == ids[None, :]).astype(mx.int32)
    slots = mx.arange(ids.shape[0])[None, :]
    pos = (matches * slots).sum(1)
    pos = mx.where(matches.any(1), pos, -1)
    self.slot_of = pos
    id_list = [int(e) for e in ids]
    w_parts = [self.disk.read(sk, id_list) for sk in self._src_weight]
    self.weight = mx.concatenate(w_parts, axis=1) if self._concat else w_parts[0]
    ngroups = self.weight.shape[-1] * 16 // self._flash_group_size
    a_parts = [self.disk.read(sk, id_list) for sk in self._src_alpha]
    s_parts = [mx.broadcast_to(a[..., None], (*a.shape, ngroups)) for a in a_parts]
    self.scales = mx.concatenate(s_parts, axis=1) if self._concat else s_parts[0]
    self.biases = -self.scales


def _flash_page(self, eid: int):
    slot = int(mx.argmin(self._age))
    id_list = [eid]
    w_parts = [self.disk.read(sk, id_list) for sk in self._src_weight]
    wrow = mx.concatenate(w_parts, axis=1) if self._concat else w_parts[0]  # (1, *row)
    ngroups = self.weight.shape[-1] * 16 // self._flash_group_size
    a_parts = [self.disk.read(sk, id_list) for sk in self._src_alpha]
    srow_full = -mx.concatenate(
        [mx.broadcast_to(a[..., None], (*a.shape, ngroups)) for a in a_parts], axis=1
    ) if self._concat else -mx.broadcast_to(a_parts[0][..., None], (*a_parts[0].shape, ngroups))
    cur = self.weight
    if slot == 0:
        self.weight = mx.concatenate([wrow, cur[1:]], axis=0)
    elif slot == cur.shape[0] - 1:
        self.weight = mx.concatenate([cur[:-1], wrow], axis=0)
    else:
        self.weight = mx.concatenate([cur[:slot], wrow, cur[slot + 1 :]], axis=0)
    cscale = self.scales
    bias_row = srow_full
    if slot == 0:
        self.scales = mx.concatenate([bias_row, cscale[1:]], axis=0)
    elif slot == cscale.shape[0] - 1:
        self.scales = mx.concatenate([cscale[:-1], bias_row], axis=0)
    else:
        self.scales = mx.concatenate([cscale[:slot], bias_row, cscale[slot + 1 :]], axis=0)
    self.biases = -self.scales
    eidx = mx.arange(self.slot_of.shape[0])
    self.slot_of = mx.where(eidx == eid, slot, self.slot_of)
    self._tick += 1
    aidx = mx.arange(self._age.shape[0])
    self._age = mx.where(aidx == slot, self._tick, self._age)


def _flash_resolve(self, indices):
    if not self.cache_enabled:
        return indices
    slot = self.slot_of[indices]
    miss = slot < 0
    if mx.any(miss):
        flat_idx = indices.reshape(-1)
        flat_miss = miss.reshape(-1)
        for j in range(flat_idx.shape[0]):
            if flat_miss[j].item():
                _flash_page(self, int(flat_idx[j].item()))
        slot = self.slot_of[indices]
    self._tick += 1
    n = self._age.shape[0]
    tick = mx.full((n,), self._tick, dtype=self._age.dtype)
    age_idx = mx.arange(n)
    hit = (age_idx[None, :] == slot.reshape(-1)[:, None]).any(0)
    self._age = mx.where(hit, tick, self._age)
    return slot


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        if getattr(self, "cache_enabled", False):
            indices = _flash_resolve(self, indices)
            sorted_indices = False
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def init_resident(self, E: int, file_index=None, key_prefix=None, group_size=128):
        _flash_init_resident(self, E, file_index, key_prefix, group_size)

    def set_active(self, ids):
        _flash_set_active(self, ids)


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        if getattr(self, "cache_enabled", False):
            indices = _flash_resolve(self, indices)
            sorted_indices = False
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def init_resident(self, E: int, file_index=None, key_prefix=None, group_size=128):
        _flash_init_resident(self, E, file_index, key_prefix, group_size)

    def set_active(self, ids):
        _flash_set_active(self, ids)

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
