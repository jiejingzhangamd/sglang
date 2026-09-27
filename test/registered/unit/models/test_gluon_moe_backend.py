from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import (
    _validate_gluon_quant_method,
)
from sglang.srt.layers.moe.gluon_backend import GluonMoeBackend
from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Backend(GluonMoeBackend):
    def __init__(self, result=None):
        self.result = result
        self.bound = False

    def bind(self, layer, experts):
        self.bound = True

    def forward(self, hidden_states, **kwargs):
        if self.result is None:
            return hidden_states + 2
        return self.result(hidden_states)


def _moe_shell() -> DeepseekV2MoE:
    moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
    torch.nn.Module.__init__(moe)
    moe._gluon_moe_backend = None
    moe.is_deepseek_v4 = False
    moe.is_hash = False
    moe.tp_size = 8
    moe.layer_id = 7
    moe.experts = torch.nn.Identity()
    moe._shared_expert_tp1 = False
    return moe


def test_gluon_backend_is_strict_and_uses_native_finalize(monkeypatch):
    moe = _moe_shell()
    monkeypatch.setattr(deepseek_v2, "get_moe_runner_backend", lambda: _Gluon())
    monkeypatch.setattr(deepseek_v2, "post_experts_all_reduce", lambda value: value + 3)
    backend = _Backend()

    moe.bind_gluon_moe_backend(backend)
    output = moe.forward_normal(torch.zeros((2, 4), dtype=torch.bfloat16))

    assert backend.bound
    torch.testing.assert_close(output, torch.full_like(output, 5))


def test_gluon_backend_missing_implementation_raises(monkeypatch):
    moe = _moe_shell()
    monkeypatch.setattr(deepseek_v2, "get_moe_runner_backend", lambda: _Gluon())

    with pytest.raises(RuntimeError, match="no Gluon implementation was bound"):
        moe.forward_normal(torch.zeros((2, 4), dtype=torch.bfloat16))


def test_gluon_backend_is_selected_before_other_forward_paths(monkeypatch):
    moe = _moe_shell()
    monkeypatch.setattr(deepseek_v2, "get_moe_runner_backend", lambda: _Gluon())

    def reject_mega_moe(*_args, **_kwargs):
        raise AssertionError("MegaMoE selection must not run for Gluon")

    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe.should_use_mega_moe",
        reject_mega_moe,
    )

    with pytest.raises(RuntimeError, match="no Gluon implementation was bound"):
        moe.forward(torch.zeros((2, 4), dtype=torch.bfloat16))


def test_gluon_backend_rejects_invalid_output(monkeypatch):
    moe = _moe_shell()
    monkeypatch.setattr(deepseek_v2, "get_moe_runner_backend", lambda: _Gluon())
    moe.bind_gluon_moe_backend(_Backend(lambda value: value[:, :2]))

    with pytest.raises(RuntimeError, match="must match the input tensor contract"):
        moe.forward_normal(torch.zeros((2, 4), dtype=torch.bfloat16))


class _Gluon:
    @staticmethod
    def is_gluon():
        return True

    @staticmethod
    def is_auto():
        return False


def test_gluon_backend_rejects_fp8_quant_method(monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "get_moe_runner_backend", lambda: _Gluon())

    with pytest.raises(ValueError, match="only serialized Quark W4A4"):
        _validate_gluon_quant_method(object(), object())


def test_gluon_backend_accepts_serialized_quark_mxfp4(monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
    from sglang.srt.layers.quantization.quark.quark import QuarkFusedMoEMethod
    from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
        QuarkW4A4MXFp4MoE,
    )

    monkeypatch.setattr(fused_moe_layer, "get_moe_runner_backend", lambda: _Gluon())
    scheme = object.__new__(QuarkW4A4MXFp4MoE)
    scheme.is_checkpoint_mxfp4_serialized = True
    layer = type("Layer", (), {"scheme": scheme})()
    quant_method = object.__new__(QuarkFusedMoEMethod)

    _validate_gluon_quant_method(layer, quant_method)


def test_gluon_backend_accepts_audited_glm_nextn_bf16_experts(monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    monkeypatch.setattr(fused_moe_layer, "get_moe_runner_backend", lambda: _Gluon())
    w13 = type("Weight", (), {"dtype": torch.bfloat16, "shape": (256, 512, 6144)})()
    w2 = type("Weight", (), {"dtype": torch.bfloat16, "shape": (256, 6144, 256)})()
    layer = type(
        "Layer",
        (),
        {
            "layer_name": "model.decoder.mlp.experts",
            "num_experts": 256,
            "hidden_size": 6144,
            "top_k": 8,
            "moe_tp_size": 8,
            "intermediate_size_per_partition": 256,
            "w13_weight": w13,
            "w2_weight": w2,
        },
    )()

    _validate_gluon_quant_method(layer, object.__new__(UnquantizedFusedMoEMethod))


def test_gluon_backend_owns_quark_mxfp4_weight_layout(monkeypatch):
    from sglang.srt.layers.moe import utils as moe_utils
    from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
        QuarkW4A4MXFp4MoE,
    )

    monkeypatch.setattr(moe_utils, "get_moe_runner_backend", lambda: _Gluon())
    scheme = object.__new__(QuarkW4A4MXFp4MoE)

    scheme.create_moe_runner(layer=object(), moe_runner_config=object())

    assert scheme.runner is None
    assert not scheme._owns_moe_runner
    assert scheme._owns_moe_weight_layout


def test_gluon_backend_accepts_deepseek_v4_serialized_fp4(monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    monkeypatch.setattr(fused_moe_layer, "get_moe_runner_backend", lambda: _Gluon())
    quant_method = object.__new__(Fp8MoEMethod)
    quant_method.is_fp4_expert = True
    quant_method.quant_config = SimpleNamespace(is_dsv4_fp4_experts=True)

    _validate_gluon_quant_method(object(), quant_method)


def _deepseek_v4_pro_backend_shell(monkeypatch):
    from sglang.srt import utils
    from sglang.srt.layers.moe.deepseek_v4_pro_gluon import (
        DeepseekV4ProGluonMoeBackend,
    )
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    monkeypatch.setattr(utils, "is_gfx95_supported", lambda: True)

    quant_method = object.__new__(Fp8MoEMethod)
    quant_method.is_fp4_expert = True
    quant_method.quant_config = SimpleNamespace(
        is_dsv4_fp4_experts=True,
        is_checkpoint_fp8_serialized=True,
        scale_fmt="ue8m0",
        weight_block_size=[128, 128],
    )
    experts = SimpleNamespace(quant_method=quant_method)
    config = SimpleNamespace(
        model_type="deepseek_v4",
        hidden_size=7168,
        n_routed_experts=384,
        num_experts_per_tok=6,
        moe_intermediate_size=3072,
        num_hidden_layers=61,
        num_hash_layers=3,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        swiglu_limit=10.0,
    )
    layer = SimpleNamespace(
        config=config,
        layer_id=3,
        tp_size=8,
        moe_ep_size=1,
        is_hash=False,
        gate=SimpleNamespace(e_score_correction_bias=object()),
        _enable_a2a_moe=False,
        alt_stream=None,
        num_fused_shared_experts=0,
        _fuse_shared_experts_inside_sbo=False,
        _shared_expert_tp1=False,
        shared_experts=object(),
        routed_scaling_factor=2.5,
    )
    backend = DeepseekV4ProGluonMoeBackend()
    return backend, layer, experts


def test_deepseek_v4_pro_backend_binds_exact_contract(monkeypatch):
    backend, layer, experts = _deepseek_v4_pro_backend_shell(monkeypatch)

    backend.bind(layer, experts)

    assert backend.layer is layer
    assert backend.experts is experts


def test_deepseek_v4_pro_backend_rejects_other_variants(monkeypatch):
    backend, layer, experts = _deepseek_v4_pro_backend_shell(monkeypatch)
    layer.config.hidden_size = 5120

    with pytest.raises(RuntimeError, match="hidden size 7168"):
        backend.bind(layer, experts)


def test_deepseek_v4_pro_backend_rejects_unsupported_decode_shape(monkeypatch):
    backend, layer, experts = _deepseek_v4_pro_backend_shell(monkeypatch)
    backend.bind(layer, experts)

    with pytest.raises(RuntimeError, match="supports only c=1 decode shapes"):
        backend.forward(torch.empty((2, 7168), dtype=torch.bfloat16))
