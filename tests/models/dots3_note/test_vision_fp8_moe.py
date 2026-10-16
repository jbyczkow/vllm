# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dots3 vision block-FP8 MoE: weight cast and fused FP8 MoE numerics."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")

DEVICE = "xpu"


def _deepgemm_per_block_cast_to_fp8(x: torch.Tensor, gran_k: int):
    """DeepGEMM ``deep_gemm.utils.math.per_block_cast_to_fp8(use_ue8m0=False)``."""
    m, n = x.shape
    pm = (m + gran_k - 1) // gran_k * gran_k
    pn = (n + gran_k - 1) // gran_k * gran_k
    x_padded = torch.zeros((pm, pn), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, gran_k, pn // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return (
        x_scaled.view_as(x_padded)[:m, :n].contiguous(),
        sf.view(x_view.size(0), x_view.size(2)),
    )


@pytest.mark.parametrize("shape", [(2112, 1536), (1536, 2112), (100, 300)])
@pytest.mark.parametrize("block_size", [64, 128])
def test_per_block_cast_matches_deepgemm(shape, block_size):
    from vllm.models.dots3_note.nvidia.vision import _per_block_cast_to_fp8_padded

    torch.manual_seed(0)
    weight = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    weight[0, 0] = 0.0
    weight[-1] *= 1e-6

    q, s = _per_block_cast_to_fp8_padded(weight, block_size)

    padded = torch.zeros(
        (
            (shape[0] + block_size - 1) // block_size * block_size,
            (shape[1] + block_size - 1) // block_size * block_size,
        ),
        dtype=weight.dtype,
        device=DEVICE,
    )
    padded[: shape[0], : shape[1]] = weight
    ref_q, ref_s = _deepgemm_per_block_cast_to_fp8(padded, block_size)

    assert q.dtype == torch.float8_e4m3fn
    assert q.shape == ref_q.shape and s.shape == ref_s.shape
    torch.testing.assert_close(s, ref_s, atol=0, rtol=0)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))


def _quant_dequant_per_token_group(x: torch.Tensor, group: int = 128):
    xf = x.float().view(x.shape[0], -1, group)
    s = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / 448.0
    xq = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (xq.float() * s).view(x.shape)


def _dequant_block(w: torch.Tensor, s: torch.Tensor, block: int = 128):
    s = s.repeat_interleave(block, dim=-2).repeat_interleave(block, dim=-1)
    return w.float() * s[..., : w.shape[-2], : w.shape[-1]]


def _ref_fused_moe_fp8(x, w13, w2, topk_weights, topk_ids, w13_s, w2_s):
    """Torch emulation of note_vision_fused_moe_fp8 (same quantization points)."""
    xq = _quant_dequant_per_token_group(x)
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for k in range(topk_ids.shape[1]):
        for e in topk_ids[:, k].unique().tolist():
            rows = (topk_ids[:, k] == e).nonzero().squeeze(-1)
            h = (xq[rows] @ _dequant_block(w13[e], w13_s[e]).T).bfloat16()
            g, u = h.chunk(2, dim=-1)
            a = (F.silu(g.float()) * u.float()).bfloat16()
            o = _quant_dequant_per_token_group(a) @ _dequant_block(w2[e], w2_s[e]).T
            out[rows] += o * topk_weights[rows, k : k + 1].float()
    return out


@pytest.mark.parametrize("num_routed", [4, 8])
@pytest.mark.parametrize("num_tokens", [1, 256, 1031])
@torch.inference_mode()
def test_fused_moe_fp8_matches_emulation(num_routed, num_tokens):
    from vllm.models.dots3_note.nvidia.vision import (
        DotsMoEVitConfig,
        MoESwiGLUFFNFP8,
    )
    from vllm.models.dots3_note.nvidia.vision_moe import note_vision_fused_moe_fp8
    from vllm.utils.deep_gemm import calc_diff

    torch.manual_seed(0)
    config = DotsMoEVitConfig(
        embed_dim=1536,
        moe_intermediate_size=2112,
        pyramid_num_routed=[num_routed],
        capacity_factor=2,
        router_scoring_func="sigmoid",
        use_bias=False,
    )
    moe = MoESwiGLUFFNFP8(config, layer_number=0).to(DEVICE, torch.bfloat16)
    moe.process_weights_after_loading()

    x = torch.randn(num_tokens, 1536, dtype=torch.bfloat16, device=DEVICE)
    topk_ids = torch.stack(
        [torch.randperm(num_routed, device=DEVICE)[:2] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand(num_tokens, 2, dtype=torch.float32, device=DEVICE)
    args = (
        x,
        moe._fused_w13_fp8,
        moe._fused_w2_fp8,
        topk_weights,
        topk_ids,
        moe._fused_w13_scale,
        moe._fused_w2_scale,
    )

    out = note_vision_fused_moe_fp8(*args)
    ref = _ref_fused_moe_fp8(*args)

    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
    assert calc_diff(out, ref) < 1e-4


@pytest.mark.parametrize("num_routed", [4, 8])
@pytest.mark.parametrize("num_tokens", [1, 256, 1031])
@torch.inference_mode()
def test_fp8_moe_close_to_bf16_moe(num_routed, num_tokens):
    from vllm.models.dots3_note.nvidia.vision import (
        DotsMoEVitConfig,
        MoESwiGLUFFN,
        MoESwiGLUFFNFP8,
    )
    from vllm.utils.deep_gemm import calc_diff

    torch.manual_seed(0)
    config = DotsMoEVitConfig(
        embed_dim=1536,
        moe_intermediate_size=2112,
        pyramid_num_routed=[num_routed],
        capacity_factor=2,
        router_scoring_func="sigmoid",
        use_bias=False,
    )
    ref = MoESwiGLUFFN(config, layer_number=0).to(DEVICE, torch.bfloat16)
    fp8 = MoESwiGLUFFNFP8(config, layer_number=0).to(DEVICE, torch.bfloat16)
    fp8.load_state_dict(ref.state_dict())
    fp8.process_weights_after_loading()

    x = torch.randn(num_tokens, 1536, dtype=torch.bfloat16, device=DEVICE)
    out = fp8(x)
    ref_out = ref(x)

    assert out.shape == ref_out.shape and out.dtype == ref_out.dtype
    assert torch.isfinite(out).all()
    # Coarse bound: FP8 weight and activation quantization noise vs bf16.
    assert calc_diff(out, ref_out) < 5e-3
