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
        self._maps = {}

    def _mmap(self, fp):
        import mmap as _mmap

        m = self._maps.get(fp)
        if m is None:
            with open(fp, "rb") as f:
                m = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
            self._maps[fp] = m
        return m

    def read(self, sf_key, ids):
        fp, abs_off, dt_str, shape = self.index[sf_key]
        np_dt, mlx_dt = _SAFETENSORS_DTYPE[dt_str]
        row_elems = int(np.prod(shape[1:]))
        row_bytes = row_elems * np.dtype(np_dt).itemsize
        mm = self._mmap(fp)
        out = np.empty((len(ids), *shape[1:]), dtype=np_dt)
        for i, e in enumerate(ids):
            off = abs_off + int(e) * row_bytes
            row = np.frombuffer(mm, dtype=np_dt, count=row_elems, offset=off)
            out[i].reshape(-1)[:] = row
        if mlx_dt == mx.bfloat16:
            # Reinterpret the 16-bit patterns as bfloat16 (bf16 == top half of
            # float32) rather than casting the integer bit values.
            out = (out.astype(np.uint32) << 16).view(np.float32)
            return mx.array(out, dtype=mx.float32).astype(mx.bfloat16)
        return mx.array(out, dtype=mlx_dt)


class ExpertSlotCache:
    """FreeToken-style global LRU expert cache shared across MoE layers.

    One slot pool serves every layer: a slot holds one expert's fused up+gate
    row *and* its down row (same slot id, two banks with different out dims).
    Routing rewrites expert ids to slot ids via ``slot_for_id``; ``resolve``
    pages in missing experts from disk, evicting the least-recently-used slot
    (timestamp LRU). Hits touch no disk at all, so recurring experts stay hot
    across tokens and layers.

    The cache owns the disk holder; per layer it knows the fused up_gate and
    down source key sets. Both projections of a layer share the same routing,
    so paging in one (layer, expert) fills both banks.
    """

    def __init__(self, num_layers: int, num_experts: int, num_slots: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.disk = _DiskHolder()
        self.disk.index = None
        # (layer, expert) -> slot ; slot -> (layer, expert) ; LRU timestamps.
        self.slot_for_id = mx.full((num_layers * num_experts,), -1, dtype=mx.int32)
        self.id_of_slot = mx.full((num_slots,), -1, dtype=mx.int32)
        self.usage = mx.zeros((num_slots,), dtype=mx.int32)
        self._step = 0
        # Per-layer source keys, set by register_layer.
        self._src = {}
        # Banks: name -> (weight, scales, biases) mx arrays of (num_slots, ...).
        self._bank = {}
        self._ngroups = {}
        self._call_cache = {}

    def set_file_index(self, file_index):
        self.disk.index = file_index

    def register_layer(self, layer_id, up_keys, up_alpha_keys, down_keys, down_alpha_keys):
        self._src[layer_id] = (
            up_keys,
            up_alpha_keys,
            down_keys,
            down_alpha_keys,
        )

    def register_bank(self, name, out_dim, packed, dtype, group_size):
        ngroups = packed * 16 // group_size
        self._ngroups[name] = (ngroups, group_size)
        self._bank[name] = (
            mx.zeros((self.num_slots, out_dim, packed), dtype=dtype),
            mx.zeros((self.num_slots, out_dim, ngroups), dtype=mx.bfloat16),
            mx.zeros((self.num_slots, out_dim, ngroups), dtype=mx.bfloat16),
        )

    def bank(self, name):
        return self._bank[name]

    def _fill_slot(self, layer_id, eid, slot):
        up_keys, up_alpha, down_keys, down_alpha = self._src[layer_id]
        ngroups_up, _ = self._ngroups["up_gate"]
        ngroups_dn, _ = self._ngroups["down"]

        w_up = [self.disk.read(k, [eid]) for k in up_keys]
        w_up = mx.concatenate(w_up, axis=1) if len(w_up) > 1 else w_up[0]
        a_up = [self.disk.read(k, [eid]) for k in up_alpha]
        a_up = mx.concatenate(a_up, axis=1) if len(a_up) > 1 else a_up[0]
        s_up = mx.broadcast_to(a_up[..., None], (*a_up.shape, ngroups_up))

        w_dn = [self.disk.read(k, [eid]) for k in down_keys]
        w_dn = mx.concatenate(w_dn, axis=1) if len(w_dn) > 1 else w_dn[0]
        a_dn = [self.disk.read(k, [eid]) for k in down_alpha]
        a_dn = mx.concatenate(a_dn, axis=1) if len(a_dn) > 1 else a_dn[0]
        s_dn = mx.broadcast_to(a_dn[..., None], (*a_dn.shape, ngroups_dn))

        w_bank, s_bank, b_bank = self._bank["up_gate"]
        w_bank[slot] = w_up[0]
        s_bank[slot] = s_up[0]
        b_bank[slot] = -s_up[0]
        w_bank, s_bank, b_bank = self._bank["down"]
        w_bank[slot] = w_dn[0]
        s_bank[slot] = s_dn[0]
        b_bank[slot] = -s_dn[0]

    def _victim(self):
        return int(mx.argmin(self.usage).item())

    def _evict(self, slot):
        fid = int(self.id_of_slot[slot].item())
        if fid >= 0:
            self.slot_for_id[fid] = -1
        self.id_of_slot[slot] = -1

    def _page_in(self, layer_id, eid):
        slot = self._victim()
        self._evict(slot)
        self._fill_slot(layer_id, eid, slot)
        fid = layer_id * self.num_experts + eid
        self.id_of_slot[slot] = fid
        self.slot_for_id[fid] = slot
        self.usage[slot] = self._step
        return slot

    def resolve(self, layer_id, expert_ids):
        flat = expert_ids.reshape(-1)
        if layer_id == 0:
            self._call_cache = {}
        key = (layer_id, int(flat[0]) if flat.size else -1)
        cached = self._call_cache.pop(key, None)
        if cached is not None and cached[0].size == flat.size:
            self._step += 1
            slots = cached[0]
            self.usage[slots] = self._step
            return slots.reshape(expert_ids.shape)
        base = layer_id * self.num_experts
        slots = self.slot_for_id[base + flat]
        self._step += 1
        mx.eval(slots)
        slots_np = np.asarray(slots.tolist(), dtype=np.int32).reshape(-1)
        if np.any(slots_np < 0):
            flat_np = np.asarray(flat.tolist(), dtype=np.int32).reshape(-1)
            for i in range(flat_np.shape[0]):
                if slots_np[i] < 0:
                    e = int(flat_np[i])
                    fid = base + e
                    slot = int(self.slot_for_id[fid].item())
                    if slot < 0:
                        slot = self._page_in(layer_id, e)
                    slots_np[i] = slot
            slots = mx.array(slots_np, dtype=mx.int32)
            mx.eval(
                self.slot_for_id, self.id_of_slot, self.usage,
                *self._bank["up_gate"], *self._bank["down"],
            )
        self.usage[slots] = self._step
        self._call_cache = {key: (slots, None)}
        return slots.reshape(expert_ids.shape)

    def reset(self):
        self.slot_for_id = mx.full((self.num_layers * self.num_experts,), -1, dtype=mx.int32)
        self.id_of_slot = mx.full((self.num_slots,), -1, dtype=mx.int32)
        self.usage = mx.zeros((self.num_slots,), dtype=mx.int32)
        self._step = 0


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
        if getattr(self, "expert_cache", None) is not None:
            indices = self.expert_cache.resolve(self.layer_id, indices)
            w, s, b = self.expert_cache.bank(self.bank_name)
            return mx.gather_qmm(
                x,
                w,
                s,
                b,
                rhs_indices=indices,
                transpose=True,
                group_size=self.expert_cache._ngroups[self.bank_name][1],
                bits=self.bits,
                mode=self.mode,
                sorted_indices=sorted_indices,
            )
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

    def attach_cache(self, cache, layer_id: int, bank_name: str):
        self.expert_cache = cache
        self.layer_id = layer_id
        self.bank_name = bank_name
        # Drop the per-layer (num_experts, ...) tensors so load-time eval does
        # not materialize the whole model; the cache banks hold resident rows.
        for k in ("weight", "scales", "biases", "bias"):
            if k in self:
                self.__delattr__(k)

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
