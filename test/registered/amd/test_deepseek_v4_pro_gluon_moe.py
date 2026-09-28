"""gfx950 coverage for the DeepSeek-V4 Pro c=1 Gluon MoE kernel."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip and is_gfx95_supported(),
    "requires AMD gfx95",
)
class TestDeepseekV4ProGluonMoe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda", 0)
        cls.router = torch.randn(384, 7168, device=cls.device, dtype=torch.bfloat16)
        cls.bias = torch.randn(384, device=cls.device, dtype=torch.float32)
        # The kernel consumes AITER's shuffled byte layout. Zero FP4 weights
        # make layout-independent compile/safety coverage while retaining the
        # exact production allocation and padded UE8M0 scale geometry.
        cls.w13 = torch.zeros(384, 768, 3584, device=cls.device, dtype=torch.uint8)
        cls.s13 = torch.full((384, 768, 224), 127, device=cls.device, dtype=torch.uint8)
        cls.w2 = torch.zeros(384, 7168, 192, device=cls.device, dtype=torch.uint8)
        cls.s2 = torch.full((384, 7168, 16), 127, device=cls.device, dtype=torch.uint8)
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4_dtype is not None:
            cls.w13 = cls.w13.view(fp4_dtype)
            cls.w2 = cls.w2.view(fp4_dtype)

    def test_target_and_mtp_shapes_compile_and_run(self):
        from sglang.srt.layers.moe.gluon_kernels.deepseek_v4_pro_tp8 import (
            fused_moe,
        )

        for tokens in (1, 4, 6):
            with self.subTest(tokens=tokens):
                hidden_states = torch.randn(
                    tokens, 7168, device=self.device, dtype=torch.bfloat16
                )
                output = fused_moe(
                    hidden_states,
                    self.router,
                    self.bias,
                    self.w13,
                    self.s13,
                    self.w2,
                    self.s2,
                )
                torch.cuda.synchronize()
                self.assertEqual(output.shape, hidden_states.shape)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(output.float()).all())

    def test_ep4_target_and_mtp_shapes_compile_and_run(self):
        from sglang.srt.layers.moe.gluon_kernels.deepseek_v4_pro_tp8 import (
            fused_moe,
        )

        # TP8/EP4 has 96 local experts and moe_tp_size=2, hence I=1536.
        w13 = torch.zeros(96, 3072, 3584, device=self.device, dtype=torch.uint8)
        s13 = torch.full((96, 3072, 224), 127, device=self.device, dtype=torch.uint8)
        w2 = torch.zeros(96, 7168, 768, device=self.device, dtype=torch.uint8)
        s2 = torch.full((96, 7168, 48), 127, device=self.device, dtype=torch.uint8)
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4_dtype is not None:
            w13 = w13.view(fp4_dtype)
            w2 = w2.view(fp4_dtype)
        for tokens in (1, 4, 6):
            with self.subTest(tokens=tokens):
                hidden_states = torch.randn(
                    tokens, 7168, device=self.device, dtype=torch.bfloat16
                )
                output = fused_moe(
                    hidden_states,
                    self.router,
                    self.bias,
                    w13,
                    s13,
                    w2,
                    s2,
                    expert_start=192,
                )
                torch.cuda.synchronize()
                self.assertEqual(output.shape, hidden_states.shape)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(output.float()).all())

    def test_sqrtsoftplus_top6_matches_torch(self):
        from sglang.srt.layers.moe.gluon_kernels.deepseek_v4_pro_tp8 import (
            _select_routes,
        )

        torch.manual_seed(11)
        tokens, splits = 6, 14
        logits = torch.randn(
            tokens, splits, 512, device=self.device, dtype=torch.float32
        )
        logits[:, :, 384:] = 0
        bias = torch.randn(384, device=self.device, dtype=torch.float32)
        ids = torch.empty(tokens * 9, device=self.device, dtype=torch.int32)
        weights = torch.empty(tokens * 9, device=self.device, dtype=torch.float32)

        _select_routes[tokens,](
            logits,
            bias,
            ids,
            weights,
            None,
            splits,
            False,
            2.5,
            num_warps=1,
        )
        torch.cuda.synchronize()

        probabilities = torch.sqrt(
            torch.nn.functional.softplus(logits[:, :, :384].sum(dim=1))
        )
        ref_ids = torch.topk(probabilities + bias, 6, dim=-1).indices
        ref_weights = torch.gather(probabilities, 1, ref_ids)
        ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True) * 2.5
        actual_ids = ids.view(tokens, 9)[:, :6]
        actual_weights = weights.view(tokens, 9)[:, :6]

        self.assertTrue(
            torch.equal(
                torch.sort(actual_ids, dim=1).values,
                torch.sort(ref_ids, dim=1).values,
            )
        )
        for token in range(tokens):
            actual = {
                int(actual_ids[token, rank]): actual_weights[token, rank]
                for rank in range(6)
            }
            reordered = torch.stack(
                [actual[int(ref_ids[token, rank])] for rank in range(6)]
            )
            torch.testing.assert_close(
                reordered, ref_weights[token], rtol=1e-5, atol=1e-6
            )
        self.assertTrue((weights.view(tokens, 9)[:, 6:] == 0).all())

    def test_ep4_route_ids_are_rank_local_and_nonlocal_weights_are_zero(self):
        from sglang.srt.layers.moe.gluon_kernels.deepseek_v4_pro_tp8 import (
            _select_routes,
        )

        logits = torch.zeros(1, 1, 512, device=self.device, dtype=torch.float32)
        bias = torch.full((384,), -100.0, device=self.device, dtype=torch.float32)
        bias[192:198] = torch.arange(6, 0, -1, device=self.device)
        ids = torch.empty(9, device=self.device, dtype=torch.int32)
        weights = torch.empty(9, device=self.device, dtype=torch.float32)

        _select_routes[1,](
            logits,
            bias,
            ids,
            weights,
            None,
            1,
            False,
            2.5,
            192,
            96,
            num_warps=1,
        )
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(
                torch.sort(ids[:6]).values,
                torch.arange(6, device=self.device, dtype=torch.int32),
            )
        )
        self.assertTrue((weights[:6] > 0).all())

        _select_routes[1,](
            logits,
            bias,
            ids,
            weights,
            None,
            1,
            False,
            2.5,
            0,
            96,
            num_warps=1,
        )
        torch.cuda.synchronize()
        self.assertTrue((ids[:6] == 96).all())
        self.assertTrue((weights[:6] == 0).all())


if __name__ == "__main__":
    unittest.main()
