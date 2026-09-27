import pytest
import torch

from sglang.srt.layers.moe.gluon_backend import GluonMoeBackend
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    _validate_gluon_quant_method,
)
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


def test_gluon_backend_rejects_fp8_quant_method(monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "get_moe_runner_backend", lambda: _Gluon())

    with pytest.raises(ValueError, match="only serialized Quark W4A4 MXFP4"):
        _validate_gluon_quant_method(object(), object())


def test_gluon_backend_accepts_serialized_quark_mxfp4(monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
    from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
        QuarkW4A4MXFp4MoE,
    )
    from sglang.srt.layers.quantization.quark.quark import QuarkFusedMoEMethod

    monkeypatch.setattr(fused_moe_layer, "get_moe_runner_backend", lambda: _Gluon())
    scheme = object.__new__(QuarkW4A4MXFp4MoE)
    scheme.is_checkpoint_mxfp4_serialized = True
    layer = type("Layer", (), {"scheme": scheme})()
    quant_method = object.__new__(QuarkFusedMoEMethod)

    _validate_gluon_quant_method(layer, quant_method)
