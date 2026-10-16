# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Absorbed-MLA (``q_v``) attention through ``xpu_ops.flash_attn_varlen_func``.

``scores = (q @ k^T + q_v @ v^T) * scale``, ``out = softmax(scores) @ v``.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")

DEVICE = "xpu"
D_QK = 64
D_V = 512


def _ref_mla_qv(q, q_v, k, v, q_lens, kv_lens, block_table, scale, causal):
    page_size = k.shape[1]
    heads_per_kv = q.shape[1] // k.shape[2]
    outs = []
    q_begin = 0
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        n = torch.arange(kv_len, device=q.device)
        blk = block_table[b, n // page_size].long()
        off = n % page_size
        kr = k[blk, off].float().repeat_interleave(heads_per_kv, dim=1)
        vr = v[blk, off].float().repeat_interleave(heads_per_kv, dim=1)
        qs = q[q_begin : q_begin + q_len].float()
        qvs = q_v[q_begin : q_begin + q_len].float()
        s = (
            torch.einsum("qhd,khd->hqk", qs, kr) + torch.einsum("qhd,khd->hqk", qvs, vr)
        ) * scale
        if causal:
            qi = torch.arange(q_len, device=q.device).view(q_len, 1)
            mask = n.view(1, kv_len) < kv_len - (q_len - 1 - qi)
            s = s.masked_fill(~mask, float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), vr))
        q_begin += q_len
    return torch.cat(outs, dim=0)


@pytest.mark.parametrize("num_heads", [(16, 1), (128, 1)])
@pytest.mark.parametrize("block_size", [1, 64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("softmax_scale", [None, 0.07])
@torch.inference_mode()
def test_xpu_ops_forwards_q_v(num_heads, block_size, causal, softmax_scale):
    from vllm._xpu_ops import xpu_ops

    torch.manual_seed(0)
    dtype = torch.bfloat16
    num_q_heads, num_kv_heads = num_heads
    q_lens = [1, 5, 37, 3]
    kv_lens = [1, 40, 37, 300]
    num_blocks = 4096 // block_size
    max_blocks = max((kv + block_size - 1) // block_size for kv in kv_lens)

    # k / v are views into one MLA cache row ([v | k]), as in the sparse backend.
    cache = torch.randn(
        num_blocks, block_size, num_kv_heads, D_V + D_QK, dtype=dtype, device=DEVICE
    )
    v = cache[..., :D_V]
    k = cache[..., D_V:]
    total_q = sum(q_lens)
    q = torch.randn(total_q, num_q_heads, D_QK, dtype=dtype, device=DEVICE)
    q_v = torch.randn(total_q, num_q_heads, D_V, dtype=dtype, device=DEVICE)
    block_table = torch.stack(
        [torch.randperm(num_blocks, device=DEVICE)[:max_blocks] for _ in q_lens]
    ).to(torch.int32)
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE
    )
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)

    out = xpu_ops.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max_blocks * block_size,
        softmax_scale=softmax_scale,
        causal=causal,
        block_table=block_table,
        seqused_k=seqused_k,
        q_v=q_v,
    )

    # FA3 q_v convention for the default scale.
    scale = (D_QK + D_V) ** -0.5 if softmax_scale is None else softmax_scale
    ref = _ref_mla_qv(q, q_v, k, v, q_lens, kv_lens, block_table, scale, causal)
    assert out.shape == (total_q, num_q_heads, D_V)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
