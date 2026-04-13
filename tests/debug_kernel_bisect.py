"""Binary search: compile the REAL conv1d kernel with different configs.

All 8 minimal pattern tests passed — the SIGSEGV is caused by the overall
complexity of the ~380-line kernel. This test compiles the actual kernel
with progressively more features enabled to find the tipping point.

Run on server:
  rm -rf ~/.triton/cache && python tests/debug_kernel_bisect.py
"""
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.mamba.causal_conv1d import (
    _causal_conv1d_fwd_kernel_npu,
    compute_conv1d_grid_npu,
)

DEVICE = "npu"


def test_real_kernel(label: str, dim: int, block_n: int, is_apc: bool):
    """Compile & run the real kernel with given config."""
    B = 2
    W = 4
    state_len = W - 1  # 3
    total_tokens = 16  # 2 seqs of 8 tokens each
    pool_size = 8
    BLOCK_M = 8

    x = torch.randn(total_tokens, dim, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(dim, W, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty_like(x)

    # conv_states: (N, dim, state_len) with stride(1)==1
    # Create (N, state_len, dim) contiguous then transpose
    cs_raw = torch.randn(
        pool_size, state_len, dim, device=DEVICE, dtype=torch.bfloat16)
    conv_states = cs_raw.transpose(1, 2)  # strides: (sl*dim, 1, dim)

    query_start_loc = torch.tensor(
        [0, 8, 16], dtype=torch.int32, device=DEVICE)
    has_initial = torch.tensor([1, 1], dtype=torch.int32, device=DEVICE)

    batch_ptr, chunk_offset_ptr, num_programs = compute_conv1d_grid_npu(
        query_start_loc, BLOCK_M, -1, x.device)

    grid = (num_programs, triton.cdiv(dim, block_n))

    # APC tensors (always provide valid tensors; guarded by IS_APC_ENABLED)
    cache_indices = torch.zeros(
        B, 3, dtype=torch.int32, device=DEVICE)
    for i in range(B):
        for j in range(3):
            cache_indices[i, j] = i * 3 + j
    first_sched = torch.zeros(B, dtype=torch.int32, device=DEVICE)
    last_sched = torch.ones(B, dtype=torch.int32, device=DEVICE)
    init_state_idx = torch.zeros(B, dtype=torch.int32, device=DEVICE)
    num_computed = torch.zeros(B, dtype=torch.int32, device=DEVICE)

    block_size = 16 if is_apc else BLOCK_M

    _causal_conv1d_fwd_kernel_npu[grid](
        x, weight, bias, conv_states, cache_indices,
        has_initial, query_start_loc,
        batch_ptr, chunk_offset_ptr,
        first_sched, last_sched, init_state_idx, num_computed,
        out,
        # Dimensions
        dim, total_tokens, pool_size,
        # Strides
        x.stride(1), x.stride(0),
        weight.stride(0), weight.stride(1),
        conv_states.stride(0), conv_states.stride(1), conv_states.stride(2),
        cache_indices.stride(0),
        out.stride(1), out.stride(0),
        block_size // BLOCK_M,
        # Others
        -1,
        # Meta-parameters
        HAS_BIAS=True,
        SILU_ACTIVATION=True,
        IS_APC_ENABLED=is_apc,
        USE_PAD_SLOT=True,
        NP2_STATELEN=4,
        BLOCK_M=BLOCK_M,
        BLOCK_N=block_n,
    )
    torch.npu.synchronize()
    print(f"✅ {label}: PASSED")


if __name__ == "__main__":
    tests = [
        # (label, dim, block_n, is_apc)
        ("T1: small dim=64, BN=64, APC=off",   64,  64,  False),
        ("T2: small dim=64, BN=64, APC=on",    64,  64,  True),
        ("T3: prod  dim=64, BN=256, APC=off",  64,  256, False),
        ("T4: prod  dim=64, BN=256, APC=on",   64,  256, True),
        ("T5: full  dim=256, BN=256, APC=off",  256, 256, False),
        ("T6: full  dim=256, BN=256, APC=on",   256, 256, True),
    ]

    for label, dim, bn, apc in tests:
        print(f"\n--- {label} ---")
        try:
            test_real_kernel(label, dim, bn, apc)
        except Exception as e:
            print(f"❌ {label}: {type(e).__name__}")
            # Print last 10 lines of error
            err_lines = str(e).strip().split('\n')
            for line in err_lines[-10:]:
                print(f"   {line}")
