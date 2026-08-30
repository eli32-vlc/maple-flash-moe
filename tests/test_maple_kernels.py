# Copyright © 2026 DeepGrove AI.

"""Portability self-check for Maple's Metal kernels and precision rules.

    python -m pytest tests/test_maple_kernels.py -v

Kernels have runtime fallbacks, so a kernel failure here means "the fast path
is off on this machine", not "the model is broken". A precision failure does
mean the model is wrong.
"""

import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models import maple


def _args(**kw):
    base = dict(
        num_hidden_layers=2,
        num_experts=64,
        num_experts_per_tok=8,
        hidden_size=2048,
        moe_intermediate_size=512,
        vocab_size=1024,
        layer_types=["sliding_attention", "full_attention"],
    )
    base.update(kw)
    return maple.ModelArgs(**base)


class TestMapleKernels(unittest.TestCase):
    def test_fused_router_matches_reference(self):
        """Fused router (gemv+softmax+top8) vs the pure-MLX path."""
        args = _args()
        gate = maple.MapleGate(args)
        gate.weight = (
            mx.random.normal((args.num_experts, args.hidden_size)) * 0.05
        ).astype(mx.bfloat16)
        mx.eval(gate.weight)
        x0 = (mx.random.normal((1, 1, args.hidden_size)) * 0.5).astype(mx.bfloat16)
        if not gate._probe(x0):
            self.skipTest("fused router disabled on this build")

        set_matches = 0
        trials = 32
        for i in range(trials):
            x = (mx.random.normal((1, 1, args.hidden_size)) * 0.5).astype(mx.bfloat16)
            fi, fs = gate._fused_call(x)
            # The router runs in float32 (`router_dtype: fp32`), so the
            # reference must not round the logits to bf16 either.
            ri, rs = maple.group_expert_select(
                x.astype(mx.float32) @ gate.weight.astype(mx.float32).T,
                args.num_experts_per_tok,
            )
            mx.eval(fi, fs, ri, rs)

            fused = {int(a): float(b) for a, b in zip(fi.reshape(-1), fs.reshape(-1))}
            ref = {int(a): float(b) for a, b in zip(ri.reshape(-1), rs.reshape(-1))}
            if set(fused) == set(ref):
                set_matches += 1
                for k in fused:
                    # Same experts selected -> scores must agree to ~1 ulp.
                    self.assertAlmostEqual(fused[k], ref[k], places=5)

        # Exact ties at the top-k boundary may select the other equally scored
        # expert; that is legitimate, but it must be rare.
        self.assertGreater(set_matches, trials * 0.75)

    def test_router_logits_are_float32(self):
        """Routing must not round the expert logits to bf16.

        With 256 experts the top-k boundary is routinely a near tie, and
        rounding logits of this magnitude to bf16 (spacing ~0.5 near 100)
        perturbs the renormalized scores by percent, not ulps. Both the fused
        and the fallback path are checked against a float64 computation.
        """
        args = _args(num_experts=256)
        gate = maple.MapleGate(args)
        mx.random.seed(0)
        # Logits land around 100, where bf16 has ~0.5 resolution.
        gate.weight = (
            mx.random.normal((args.num_experts, args.hidden_size)) * 0.05 + 0.05
        ).astype(mx.bfloat16)
        mx.eval(gate.weight)

        x = (mx.random.normal((1, 1, args.hidden_size)) * 0.2 + 1.0).astype(mx.bfloat16)
        w64 = np.array(gate.weight.astype(mx.float32), dtype=np.float64)
        x64 = np.array(x.astype(mx.float32), dtype=np.float64).reshape(-1)
        logits = w64 @ x64
        self.assertGreater(np.abs(logits).max(), 20.0, "test needs large logits")
        p = np.exp(logits - logits.max())
        p /= p.sum()
        top = np.argsort(-p)[: args.num_experts_per_tok]
        expect = dict(zip(top.tolist(), (p[top] / p[top].sum()).tolist()))

        for fused in (True, False):
            if fused and not gate._probe(x):
                continue
            gate._fused = fused
            inds, scores = gate(x)
            mx.eval(inds, scores)
            self.assertEqual(scores.dtype, mx.float32, "scores must stay float32")
            got = {
                int(i): float(s) for i, s in zip(inds.reshape(-1), scores.reshape(-1))
            }
            label = "fused" if fused else "fallback"
            self.assertEqual(set(got), set(expect), f"{label}: wrong experts selected")
            for e, want in expect.items():
                self.assertAlmostEqual(
                    got[e],
                    want,
                    delta=1e-3 * want,
                    msg=f"{label}: expert {e} score {got[e]} != {want}",
                )

    def test_add_rms_norm_matches_reference(self):
        """Fused residual add + RMSNorm vs the stock two-step path."""
        args = _args()
        dim, eps = args.hidden_size, args.rms_norm_eps
        w = (mx.random.normal((dim,)) * 0.1 + 1.0).astype(mx.bfloat16)
        x = (mx.random.normal((1, 1, dim)) * 0.5).astype(mx.bfloat16)
        r = (mx.random.normal((1, 1, dim)) * 0.5).astype(mx.bfloat16)
        if not maple._add_rms_norm_ok(dim, mx.bfloat16, w, eps):
            self.skipTest("fused add+norm disabled on this build")

        h, hn = maple._add_rms_norm(x, r, w, eps)
        mx.eval(h, hn)
        # The residual stream must be rounded exactly once, like a bf16 add.
        self.assertTrue(mx.array_equal(h, x + r), "residual add is not bit-exact")
        ref = mx.fast.rms_norm(
            (x + r).astype(mx.float32), w.astype(mx.float32), eps
        ).astype(mx.bfloat16)
        got = np.array(hn.astype(mx.float32))
        want = np.array(ref.astype(mx.float32))
        # bf16 has ~0.4% resolution; the two differ only in reduction order.
        self.assertTrue(
            np.allclose(got, want, rtol=8e-3), f"max |d| {np.abs(got - want).max()}"
        )

    def test_qk_norm_rope_matches_reference(self):
        """Fused per-head norm + partial RoPE vs q_norm/k_norm + mx.fast.rope,
        on both the RoPE and the NoPE layer type."""
        for layer_type in ("sliding_attention", "full_attention"):
            args = _args(num_hidden_layers=1, layer_types=[layer_type])
            attn = maple.MapleAttention(args, 0)
            n = args.num_attention_heads + args.num_key_value_heads
            attn.q_norm.weight = (
                mx.random.normal((args.head_dim,)) * 0.1 + 1.0
            ).astype(mx.bfloat16)
            attn.k_norm.weight = (
                mx.random.normal((args.head_dim,)) * 0.1 + 1.0
            ).astype(mx.bfloat16)
            qk = (mx.random.normal((n, args.head_dim)) * 0.5).astype(mx.bfloat16)
            mx.eval(attn.parameters(), qk)

            # A position well past the sliding window, so a rotation that
            # ignores the offset cannot pass.
            got, want = attn._qk_fused(qk, 613), attn._qk_reference(qk, 613)
            mx.eval(got, want)
            self.assertEqual(got.dtype, mx.bfloat16, layer_type)
            g = np.array(got.astype(mx.float32))
            w = np.array(want.astype(mx.float32))
            self.assertTrue(
                np.allclose(g, w, rtol=8e-3, atol=8e-3),
                f"{layer_type}: max |d| {np.abs(g - w).max()}",
            )

    def test_probe_rejects_a_mismatched_kernel(self):
        """The self-check must latch the fallback rather than ship garbage.

        Norm weights in a dtype the kernel was not templated for are the
        realistic case: the kernel reinterprets the buffer and the output is
        silently wrong, so only a value check catches it.
        """
        args = _args(num_hidden_layers=1, layer_types=["sliding_attention"])
        attn = maple.MapleAttention(args, 0)  # q_norm/k_norm default to float32
        qk = (mx.random.normal((20, args.head_dim)) * 0.5).astype(mx.bfloat16)
        mx.eval(attn.parameters(), qk)
        self.assertFalse(
            maple._matches(
                lambda: (attn._qk_fused(qk, 7),),
                lambda: (attn._qk_reference(qk, 7),),
            )
        )

    def test_decode_matches_with_fast_paths_off(self):
        """A decode step through the fused kernels must equal the stock path."""
        mx.random.seed(0)
        args = _args()
        model = maple.Model(args)
        for layer in model.layers:
            layer.mlp.gate.weight = (
                mx.random.normal((args.num_experts, args.hidden_size)) * 0.05
            )
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())

        def decode(fused):
            cache = model.make_cache()
            model(mx.array([[3, 1, 4, 1, 5]]), cache=cache)
            model.model._fused_add_norm = fused
            for layer in model.layers:
                layer.self_attn._fused_qk = fused
                # The router is compared against float64 separately; a top-8
                # tie flipping between paths would make this test flaky.
                layer.mlp.gate._fused = False
            out = model(mx.array([[9]]), cache=cache)
            mx.eval(out)
            return np.array(out.astype(mx.float32)).reshape(-1)

        # `None` leaves the probes to decide, i.e. exactly what a user gets.
        fast = decode(None)
        if not model.model._fused_add_norm:
            self.skipTest("fused decode disabled on this build")
        self.assertTrue(all(l.self_attn._fused_qk for l in model.layers))
        slow = decode(False)
        self.assertEqual(int(fast.argmax()), int(slow.argmax()), "argmax differs")
        self.assertTrue(
            np.allclose(fast, slow, rtol=5e-2, atol=5e-2),
            f"max |d| {np.abs(fast - slow).max()}",
        )

    def test_experts_clamp_both_swiglu_branches(self):
        """silu(min(gate, 7)) * clip(up, -7, 7), in the activation dtype."""
        gate = mx.array([[-3.0, 0.5, 9.0, 40.0]], dtype=mx.bfloat16)
        up = mx.array([[100.0, -50.0, 2.0, -0.25]], dtype=mx.bfloat16)
        got = maple.clamped_swiglu(gate, up)
        mx.eval(got)
        self.assertEqual(got.dtype, mx.bfloat16, "clamp must not promote to f32")

        g = np.array(gate.astype(mx.float32))
        u = np.array(up.astype(mx.float32))
        g = np.minimum(g, maple.MLP_CLAMP)
        u = np.clip(u, -maple.MLP_CLAMP, maple.MLP_CLAMP)
        want = (g / (1 + np.exp(-g))) * u
        self.assertTrue(
            np.allclose(np.array(got.astype(mx.float32)), want, rtol=8e-3),
            f"{np.array(got.astype(mx.float32))} != {want}",
        )

    def test_row_and_group_scale_layouts_agree(self):
        """A row_alpha checkpoint must expand to the per-group tensors."""
        args = _args(quantization={"group_size": 128, "bits": 2})
        model = maple.Model(args)

        n, k, groups = 256, 2048, 2048 // 128
        alpha = mx.abs(mx.random.normal((n,))).astype(mx.bfloat16) + 0.01
        packed = mx.random.randint(0, 2**31, (n, k // 16)).astype(mx.uint32)
        # o_proj is not part of the q/k/v fusion, so it exercises the
        # row_alpha expansion on its own.
        prefix = "model.layers.0.self_attn.o_proj"

        rows = model.sanitize(
            {f"{prefix}.weight": packed, f"{prefix}.row_alpha": alpha}
        )
        scales = mx.broadcast_to(alpha[:, None], (n, groups))
        mx.eval(rows, scales)

        self.assertIn(f"{prefix}.scales", rows)
        self.assertIn(f"{prefix}.biases", rows)
        self.assertTrue(mx.array_equal(rows[f"{prefix}.scales"], scales))
        self.assertTrue(mx.array_equal(rows[f"{prefix}.biases"], -scales))
        # `--group-scales` checkpoints must pass through untouched.
        grouped = model.sanitize(
            {f"{prefix}.weight": packed, f"{prefix}.scales": scales}
        )
        self.assertTrue(mx.array_equal(grouped[f"{prefix}.scales"], scales))


class TestExpertSlotCache(unittest.TestCase):
    """Correctness of the shared LRU expert slot cache (FreeToken-style)."""

    def _make_cache(self, num_layers=1, num_experts=4, num_slots=16, out_dim=8, packed=4):
        from mlx_lm.models.switch_layers import ExpertSlotCache

        cache = ExpertSlotCache(num_layers, num_experts, num_slots)
        cache.register_bank("up_gate", out_dim, packed, mx.uint32, group_size=128)
        cache.register_bank("down", out_dim, packed, mx.uint32, group_size=128)

        class _Disk:
            index = {}

            def read(self, key, ids):
                seed = hash(key) & 0xFFFF
                rng = np.random.default_rng(seed)
                if key.endswith(".alpha"):
                    return mx.array(
                        rng.random((len(ids), out_dim), dtype=np.float32),
                        dtype=mx.bfloat16,
                    )
                return mx.array(
                    rng.integers(0, 3, size=(len(ids), out_dim, packed), dtype=np.uint32)
                )

        cache.disk = _Disk()
        for l in range(num_layers):
            cache.register_layer(
                l,
                [f"l{l}.up.weight"],
                [f"l{l}.up.alpha"],
                [f"l{l}.down.weight"],
                [f"l{l}.down.alpha"],
            )
        return cache

    def test_resolve_pages_in_and_is_stable(self):
        cache = self._make_cache()
        ids = mx.array([1, 2, 3, 2], dtype=mx.int32)
        s1 = cache.resolve(0, ids)
        self.assertEqual(s1.tolist(), [0, 1, 2, 1])  # distinct experts -> distinct slots
        s2 = cache.resolve(0, ids)
        self.assertEqual(s1.tolist(), s2.tolist())  # stable across calls

    def test_resolve_evicts_least_recently_used(self):
        cache = self._make_cache(num_experts=4, num_slots=2)
        cache.resolve(0, mx.array([0, 1], dtype=mx.int32))
        cache.resolve(0, mx.array([2, 3], dtype=mx.int32))  # evicts 0/1
        # 0 must be re-paged in (never -1) and 2/3 must still be resident.
        s0 = cache.resolve(0, mx.array([0], dtype=mx.int32))
        self.assertNotEqual(s0.tolist(), [-1])
        s23 = cache.resolve(0, mx.array([2, 3], dtype=mx.int32))
        self.assertNotIn(-1, s23.tolist())

    def test_weights_match_source(self):
        cache = self._make_cache()
        cache.resolve(0, mx.array([1], dtype=mx.int32))
        slot = int(cache.slot_for_id[1].item())
        w_bank = cache._bank["up_gate"][0][slot]
        want = cache.disk.read("l0.up.weight", [1])[0]
        mx.eval(w_bank, want)
        self.assertTrue(mx.array_equal(w_bank, want))


if __name__ == "__main__":
    unittest.main()
