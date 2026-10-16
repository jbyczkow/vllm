# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dots3 audio encoder: packed varlen attention must match per-sequence SDPA."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")

DEVICE = "xpu"


@pytest.mark.parametrize("seq_lens", [[1500], [7, 129, 1500]])
@torch.inference_mode()
def test_audio_varlen_matches_sdpa(seq_lens):
    from vllm.models.dots3_note.nvidia.audio_encoder import WhisperAttention

    torch.manual_seed(0)
    dtype = torch.bfloat16
    attn = WhisperAttention(embed_dim=1280, num_heads=20).to(DEVICE, dtype)
    hidden = torch.randn(sum(seq_lens), 1280, dtype=dtype, device=DEVICE)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=DEVICE,
    )

    out, _ = attn.forward_flash_attn(
        hidden,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(seq_lens),
        max_seqlen_kv=max(seq_lens),
    )

    def heads(x):
        return x.view(-1, attn.num_heads, attn.head_dim).transpose(0, 1)

    refs = []
    for chunk in hidden.split(seq_lens):
        o = F.scaled_dot_product_attention(
            heads(attn.q_proj(chunk)),
            heads(attn.k_proj(chunk)),
            heads(attn.v_proj(chunk)),
            scale=attn.scaling,
        )
        refs.append(attn.out_proj(o.transpose(0, 1).reshape(-1, attn.embed_dim)))
    torch.testing.assert_close(out, torch.cat(refs), atol=2e-2, rtol=2e-2)
