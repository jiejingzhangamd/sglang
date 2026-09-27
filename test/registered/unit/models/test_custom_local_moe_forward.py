import pytest
import torch

from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _moe_shell() -> DeepseekV2MoE:
    moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
    torch.nn.Module.__init__(moe)
    moe._custom_local_moe_forward = None
    moe.is_deepseek_v4 = False
    moe.tp_size = 8
    moe._shared_expert_tp1 = False
    return moe


def test_custom_local_moe_forward_uses_native_finalize(monkeypatch):
    moe = _moe_shell()
    seen = {}

    def custom(hidden_states, **kwargs):
        seen.update(kwargs)
        return hidden_states + 2

    moe.register_custom_local_moe_forward(custom)
    monkeypatch.setattr(deepseek_v2, "post_experts_all_reduce", lambda value: value + 3)

    hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)
    output = moe.forward_normal(hidden_states, skip_shared_experts=True)

    torch.testing.assert_close(output, torch.full_like(hidden_states, 5))
    assert seen["skip_shared_experts"] is True
    assert seen["input_ids"] is None
    assert seen["input_ids_global"] is None
    assert seen["num_token_non_padded"] is None


def test_custom_local_moe_forward_registration_is_one_shot():
    moe = _moe_shell()
    moe.register_custom_local_moe_forward(lambda hidden_states, **kwargs: hidden_states)

    with pytest.raises(RuntimeError, match="already registered"):
        moe.register_custom_local_moe_forward(
            lambda hidden_states, **kwargs: hidden_states
        )


def test_custom_local_moe_forward_validates_output_contract():
    moe = _moe_shell()
    moe.register_custom_local_moe_forward(
        lambda hidden_states, **kwargs: hidden_states[:, :2]
    )

    with pytest.raises(RuntimeError, match="must match the input tensor contract"):
        moe.forward_normal(torch.zeros((2, 4), dtype=torch.bfloat16))
