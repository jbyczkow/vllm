# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the CUDA ``cp_gather_cache`` MLA tests against the XPU kernel."""

import pytest
import torch

from tests.kernels.attention import test_cache as cuda_cache_tests
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="XPU cp_gather_cache"
)

DEVICE = "xpu:0"


@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("block_size", [1, 16, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_cp_gather_cache_mla(
    kv_lora_rank, qk_rope_head_dim, block_size, dtype, kv_cache_dtype
):
    cuda_cache_tests.test_cp_gather_cache_mla(
        kv_lora_rank,
        qk_rope_head_dim,
        block_size,
        num_blocks=1024,
        max_seq_len=512,
        batch_size=8,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        device=DEVICE,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("unaligned", [False, True])
def test_cp_gather_cache_mla_with_seq_starts(kv_cache_dtype, unaligned):
    cuda_cache_tests.test_cp_gather_cache_mla_with_seq_starts(
        kv_cache_dtype, unaligned, DEVICE
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("use_seq_starts", [False, True])
def test_cp_gather_cache_mla_large_uneven_sequences(kv_cache_dtype, use_seq_starts):
    cuda_cache_tests.test_cp_gather_cache_mla_large_uneven_sequences(
        kv_cache_dtype, use_seq_starts, DEVICE
    )
