#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch_npu
from einops import rearrange
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.triton_utils import triton
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import fused_qkvzba_split_reshape_cat
from vllm_ascend.ops.triton.fla.sigmoid_gating import fused_sigmoid_gating_delta_rule_update
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_fn
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_update_npu
from vllm_ascend.utils import enable_sp


def to_int64_tuple(tensor: torch.Tensor) -> tuple:
    return tuple(tensor.to(torch.int64).tolist())


def _build_initial_state(
    ssm_state: torch.Tensor,
    attn_metadata,
    num_decodes: int,
    num_prefills: int,
    transpose_state: bool,
) -> torch.Tensor:
    """Build initial SSM state tensor from SOURCE pool slots for all-mode.

    Reads block_state_indices (SOURCE), transposes if needed, and zeroes
    sequences without cached initial state.

    Args:
        transpose_state: True when pool=[H,V,K], kernel needs [H,K,V].
    Returns:
        initial_state: [batch, H, K, V] in kernel layout.
    """
    batch = num_decodes + num_prefills
    src_slots = attn_metadata.block_state_indices[:batch].long()
    valid = src_slots >= 0

    if transpose_state:
        # Pool [N, H, V, K] → kernel [batch, H, K, V]
        H, V, K = ssm_state.shape[1], ssm_state.shape[2], ssm_state.shape[3]
        initial = ssm_state.new_zeros(batch, H, K, V)
    else:
        # Pool [N, H, K, V] = kernel [batch, H, K, V]
        initial = ssm_state.new_zeros(batch, *ssm_state.shape[1:])

    if valid.any():
        states = ssm_state[src_slots[valid]]
        if transpose_state:
            # NPU layout: pool [H, V, K] → kernel [H, K, V]
            states = states.transpose(-1, -2).contiguous()
        initial[valid] = states

    # Zero PREFILL sequences without cached initial state.
    # Decode sequences always have valid cached state (they wouldn't be
    # in decode phase otherwise), so only zero prefill entries.
    has_initial_state = attn_metadata.has_initial_state
    if has_initial_state is not None and num_prefills > 0:
        prefill_has_init = has_initial_state[num_decodes:num_decodes + num_prefills]
        initial[num_decodes:][~prefill_has_init] = 0

    return initial


def _write_final_states(
    ssm_state: torch.Tensor,
    final_state: torch.Tensor,
    attn_metadata,
    transpose_state: bool,
) -> None:
    """Write kernel final states to DEST pool slots.

    DEST = non_spec_state_indices_tensor (overridden to last-scheduled
    block's slot in all-mode).

    Args:
        transpose_state: True when kernel=[H,K,V], pool needs [H,V,K].
    """
    dest_slots = attn_metadata.non_spec_state_indices_tensor
    valid = dest_slots >= 0
    if not valid.any():
        return

    valid_slots = dest_slots[valid].long()
    valid_states = final_state[valid]

    if transpose_state:
        # kernel [H, K, V] → pool [H, V, K]
        ssm_state[valid_slots] = (
            valid_states.transpose(-1, -2).contiguous().to(ssm_state.dtype)
        )
    else:
        ssm_state[valid_slots] = valid_states.to(ssm_state.dtype)


def _scatter_intermediate_states(
    ssm_state: torch.Tensor,
    intermediate_states: torch.Tensor,
    attn_metadata,
    num_decodes: int,
    transpose_state: bool,
) -> None:
    """Scatter block-boundary SSM states from chunk history to pool.

    For each prefill sequence, extracts states at block boundaries from
    intermediate_states and writes to the corresponding pool slots via
    block_table_2d. Only processes prefill sequences (decode final states
    are handled by _write_final_states).

    Fully vectorized — no Python per-sequence loops or .item() calls
    beyond a single scalar sync for total entry count.

    Args:
        intermediate_states: [total_chunks, H, K, V] from kernel.
        transpose_state: True when kernel=[H,K,V], pool needs [H,V,K].
    """
    block_size = attn_metadata.mamba_block_size
    chunk_size = attn_metadata.all_mode_chunk_size
    chunks_per_block = block_size // chunk_size
    prefill_chunk_start = attn_metadata.prefill_chunk_start
    prefill_offsets = attn_metadata.prefill_chunk_offsets
    num_prefills = len(prefill_offsets) - 1
    if num_prefills <= 0:
        return

    dev = intermediate_states.device
    block_table = attn_metadata.block_table_2d[num_decodes:]
    first_sched = attn_metadata.block_idx_first_scheduled_token[num_decodes:]
    last_sched = attn_metadata.block_idx_last_scheduled_token[num_decodes:]

    # Per-prefill: number of intermediate blocks to scatter (exclude last block)
    n_blocks_all = last_sched - first_sched  # (num_prefills,)
    active = n_blocks_all > 0
    if not active.any():
        return

    active_idx = torch.where(active)[0]  # indices of active prefills
    active_n = n_blocks_all[active_idx]   # block counts for active prefills
    total_entries = int(active_n.sum().item())  # single scalar sync
    if total_entries == 0:
        return

    # Alignment: unaligned chunk offset per prefill
    num_computed_all = attn_metadata.num_computed_tokens_all
    unaligned_all = (num_computed_all[num_decodes:num_decodes + num_prefills] % block_size) // chunk_size

    # Flatten: for each active prefill, generate n_blocks entries
    flat_which = torch.repeat_interleave(
        torch.arange(active_idx.size(0), device=dev), active_n.int()
    )
    # k = 0..n_blocks[i]-1 within each active prefill
    entry_starts = torch.nn.functional.pad(active_n.cumsum(0)[:-1], (1, 0))
    flat_k = torch.arange(total_entries, device=dev) - entry_starts[flat_which]

    # Map back to original prefill indices
    flat_prefill = active_idx[flat_which]

    # h_indices into intermediate_states
    flat_seq_starts = prefill_chunk_start + prefill_offsets[flat_prefill]
    flat_h = (
        flat_seq_starts
        + (flat_k + 1) * chunks_per_block
        - 1
        - unaligned_all[flat_prefill]
    )

    # cache_slots from block_table
    flat_cols = first_sched[flat_prefill] + flat_k
    flat_cache_slots = block_table[flat_prefill, flat_cols]

    valid = flat_cache_slots >= 0
    if not valid.any():
        return

    write_states = intermediate_states[flat_h[valid].long()]
    if transpose_state:
        write_states = write_states.transpose(-1, -2).contiguous()
    ssm_state[flat_cache_slots[valid].long()] = write_states.to(ssm_state.dtype)






class AscendGatedDeltaNetAttention(GatedDeltaNetAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        if not self.gqa_interleaved_layout:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            num_tokens = mixed_qkvz.size(0)
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            ba, _ = self.in_proj_ba(hidden_states)
            b, a = ba.chunk(2, dim=-1)

            b = b.contiguous()
            a = a.contiguous()
        else:
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            num_tokens = projected_states_qkvz.size(0)

            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                projected_states_qkvz,
                projected_states_ba,
                triton.cdiv(self.num_k_heads, self.tp_size),
                triton.cdiv(self.num_v_heads, self.tp_size),
                self.head_k_dim,
                self.head_v_dim,
            )

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        maybe_save_kv_layer_to_connector("", [])
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation (called by custom op).
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        # All-mode prefix caching detection
        is_all_mode = getattr(attn_metadata, 'is_all_mode', False)

        # All-mode decode: pre-copy state when crossing block boundaries.
        # NPU GDN decode kernels use a single state_indices for both read
        # and write. When a decode token is the first in a new block, the
        # new block slot is uninitialized. Copy state from the previous
        # block so the kernel reads correct initial state.
        # (Not needed when num_prefills > 0: _build_initial_state reads
        # from SOURCE and _write_final_states writes to DEST natively.)
        if is_all_mode and attn_metadata.num_prefills == 0 and attn_metadata.num_decodes > 0:
            _n_dec = attn_metadata.num_decodes
            _src_slots = attn_metadata.block_state_indices[:_n_dec]
            _dst_slots = non_spec_state_indices_tensor[:_n_dec]
            _boundary_mask = _src_slots != _dst_slots
            if _boundary_mask.any():
                _bd_idx = _boundary_mask.nonzero(as_tuple=True)[0]
                self_kv_cache[0][_dst_slots[_bd_idx]] = self_kv_cache[0][_src_slots[_bd_idx]]
                self_kv_cache[1][_dst_slots[_bd_idx]] = self_kv_cache[1][_src_slots[_bd_idx]]

        if not enable_sp():
            mixed_qkv = mixed_qkv[:num_actual_tokens]
            b = b[:num_actual_tokens]
            a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update_npu(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][: attn_metadata.num_spec_decodes],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                if is_all_mode:
                    # All-mode: split decode/prefill for conv1d (upstream pattern)
                    num_decodes = attn_metadata.num_decodes
                    num_decode_tokens = attn_metadata.num_decode_tokens
                    block_table_2d = attn_metadata.block_table_2d
                    block_idx_first_sched = attn_metadata.block_idx_first_scheduled_token
                    block_idx_last_sched = attn_metadata.block_idx_last_scheduled_token
                    num_computed_all = attn_metadata.num_computed_tokens_all
                    _bs = attn_metadata.mamba_block_size
                    block_idx_last_computed = torch.clamp(
                        (num_computed_all + _bs - 1) // _bs - 1, min=0
                    )

                    # Zero conv state for sequences without cached initial state
                    no_init = ~has_initial_state
                    if no_init.any():
                        slots = attn_metadata.block_state_indices[no_init]
                        valid = slots >= 0
                        if valid.any():
                            self_kv_cache[0][slots[valid].long()] = 0

                    # Decode tokens: use existing decode kernel
                    if num_decodes > 0 and num_decode_tokens > 0:
                        mixed_qkv_decode = mixed_qkv_non_spec[:num_decode_tokens]
                        mixed_qkv_decode = causal_conv1d_update_npu(
                            mixed_qkv_decode,
                            conv_state,
                            conv_weights,
                            self.conv1d.bias,
                            self.activation,
                            conv_state_indices=block_table_2d[:num_decodes],
                            block_idx_last_scheduled_token=block_idx_last_sched[:num_decodes],
                            initial_state_idx=block_idx_last_computed[:num_decodes],
                            validate_data=True,
                        )
                    else:
                        mixed_qkv_decode = None

                    # Prefill tokens: use new fwd kernel via causal_conv1d_fn
                    num_prefills = attn_metadata.num_prefills
                    if num_prefills > 0:
                        mixed_qkv_prefill = mixed_qkv_non_spec[num_decode_tokens:]
                        # Transpose to channel-first: (tokens, dim) -> (dim, tokens)
                        mixed_qkv_prefill_T = mixed_qkv_prefill.transpose(0, 1)

                        # Slice all-mode params to prefill-only
                        has_initial_state_p = has_initial_state[num_decodes:]
                        block_table_2d_p = block_table_2d[num_decodes:]
                        block_idx_first_sched_p = block_idx_first_sched[num_decodes:]
                        block_idx_last_sched_p = block_idx_last_sched[num_decodes:]
                        block_idx_last_computed_p = block_idx_last_computed[num_decodes:]
                        num_computed_p = num_computed_all[num_decodes:]

                        # Adjust query_start_loc to prefill-only (start at 0)
                        qsl_p = non_spec_query_start_loc[num_decodes:]
                        if qsl_p[0] != 0:
                            qsl_p = qsl_p - qsl_p[0]

                        mixed_qkv_prefill_T = causal_conv1d_fn(
                            mixed_qkv_prefill_T,
                            conv_weights,
                            bias=self.conv1d.bias,
                            activation=self.activation,
                            conv_states=conv_state,
                            has_initial_state=has_initial_state_p,
                            cache_indices=block_table_2d_p,
                            query_start_loc=qsl_p.to(torch.int32),
                            block_idx_first_scheduled_token=block_idx_first_sched_p,
                            block_idx_last_scheduled_token=block_idx_last_sched_p,
                            initial_state_idx=block_idx_last_computed_p,
                            num_computed_tokens=num_computed_p,
                            block_size_to_align=_bs,
                        )
                        # Transpose back: (dim, tokens) -> (tokens, dim)
                        mixed_qkv_prefill = mixed_qkv_prefill_T.transpose(0, 1)
                    else:
                        mixed_qkv_prefill = None

                    # Concatenate decode + prefill results
                    if mixed_qkv_decode is not None and mixed_qkv_prefill is not None:
                        mixed_qkv_non_spec = torch.cat([mixed_qkv_decode, mixed_qkv_prefill], dim=0)
                    elif mixed_qkv_decode is not None:
                        mixed_qkv_non_spec = mixed_qkv_decode
                    elif mixed_qkv_prefill is not None:
                        mixed_qkv_non_spec = mixed_qkv_prefill
                else:
                    conv_weights_T = conv_weights.transpose(0, 1)
                    activation_num = 1 if self.activation else 0
                    mixed_qkv_non_spec = torch.ops._C_ascend.npu_causal_conv1d_custom(
                        mixed_qkv_non_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=to_int64_tuple(non_spec_query_start_loc),
                        cache_indices_opt=to_int64_tuple(non_spec_state_indices_tensor),
                        initial_state_mode_opt=to_int64_tuple(has_initial_state),
                        num_accepted_tokens_opt=[],
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=0,
                    )
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update_npu(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[: attn_metadata.num_actual_tokens],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

        # 2. Recurrent attention
        if self.gqa_interleaved_layout:
            # Qwen3Next: torch_npu ops support float16/bf16 ssm_state.
            # g/beta are needed for both spec-decode and decode, so compute unconditionally.
            g, beta = fused_gdn_gating_patch(self.A_log, a, b, self.dt_bias)
            if spec_sequence_masks is not None:
                if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                    g_spec = g
                    beta_spec = beta
                    g_non_spec = None
                    beta_non_spec = None
                else:
                    g_spec = g.index_select(1, spec_token_indx)
                    beta_spec = beta.index_select(1, spec_token_indx)
                    g_non_spec = g.index_select(1, non_spec_token_indx)
                    beta_non_spec = beta.index_select(1, non_spec_token_indx)
            else:
                g_spec = None
                beta_spec = None
                g_non_spec = g
                beta_non_spec = beta

            # 2.1: Process the multi-query part
            if spec_sequence_masks is not None:
                cu_seqlens = spec_query_start_loc[: attn_metadata.num_spec_decodes + 1]
                actual_seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                query_spec = l2norm_fwd(query_spec)
                key_spec = l2norm_fwd(key_spec)
                core_attn_out_spec = torch_npu.npu_recurrent_gated_delta_rule(
                    query=query_spec.squeeze(0),
                    key=key_spec.squeeze(0),
                    value=value_spec.squeeze(0),
                    g=g_spec.squeeze(0),
                    beta=beta_spec.squeeze(0),
                    state=ssm_state,
                    scale=key_spec.shape[-1] ** -0.5,
                    actual_seq_lengths=actual_seq_lengths,
                    ssm_state_indices=spec_state_indices_tensor.flatten(),
                    num_accepted_tokens=num_accepted_tokens.to(torch.int32),
                ).unsqueeze(0)
            else:
                core_attn_out_spec, last_recurrent_state = None, None

            # 2.2: Process the remaining part
            if attn_metadata.num_prefills > 0:
                if is_all_mode:
                    # All-mode: single-pass with intermediate state scatter
                    # (Qwen3Next: pool=[H,V,K], kernel=[H,K,V], transpose_state=True)
                    initial_state = _build_initial_state(
                        ssm_state, attn_metadata,
                        attn_metadata.num_decodes, attn_metadata.num_prefills,
                        transpose_state=True,
                    )
                    non_spec_chunked_prefill_meta = getattr(
                        attn_metadata, "non_spec_chunked_prefill_meta", None
                    )
                    (core_attn_out_non_spec, last_recurrent_state, intermediate_states) = (
                        chunk_gated_delta_rule(
                            q=query_non_spec,
                            k=key_non_spec,
                            v=value_non_spec,
                            g=g_non_spec,
                            beta=beta_non_spec,
                            initial_state=initial_state,
                            output_final_state=True,
                            cu_seqlens=non_spec_query_start_loc,
                            prebuilt_meta=non_spec_chunked_prefill_meta,
                            head_first=False,
                            use_qk_l2norm_in_kernel=True,
                            return_intermediate_states=True,
                            state_dtype=ssm_state.dtype,
                        )
                    )
                    _write_final_states(ssm_state, last_recurrent_state,
                                        attn_metadata, transpose_state=True)
                    if intermediate_states is not None:
                        _scatter_intermediate_states(
                            ssm_state, intermediate_states, attn_metadata,
                            attn_metadata.num_decodes, transpose_state=True,
                        )
                else:
                    initial_state = ssm_state[non_spec_state_indices_tensor].transpose(-1, -2).contiguous()
                    initial_state[~has_initial_state, ...] = 0
                    non_spec_chunked_prefill_meta = getattr(attn_metadata, "non_spec_chunked_prefill_meta", None)
                    (core_attn_out_non_spec, last_recurrent_state) = chunk_gated_delta_rule(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_non_spec,
                        beta=beta_non_spec,
                        initial_state=initial_state,
                        output_final_state=True,
                        cu_seqlens=non_spec_query_start_loc,
                        prebuilt_meta=non_spec_chunked_prefill_meta,
                        head_first=False,
                        use_qk_l2norm_in_kernel=True,
                    )
                    ssm_state[non_spec_state_indices_tensor] = (
                        last_recurrent_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
                    )
            elif attn_metadata.num_decodes > 0:
                cu_seqlens = non_spec_query_start_loc[: attn_metadata.num_decodes + 1]
                actual_seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                query_non_spec = l2norm_fwd(query_non_spec)
                key_non_spec = l2norm_fwd(key_non_spec)
                core_attn_out_non_spec = torch_npu.npu_recurrent_gated_delta_rule(
                    query=query_non_spec.squeeze(0),
                    key=key_non_spec.squeeze(0),
                    value=value_non_spec.squeeze(0),
                    g=g_non_spec.squeeze(0) if g_non_spec is not None else g_non_spec,
                    beta=beta_non_spec.squeeze(0) if beta_non_spec is not None else beta_non_spec,
                    state=ssm_state,
                    scale=key_non_spec.shape[-1] ** -0.5,
                    actual_seq_lengths=actual_seq_lengths,
                    ssm_state_indices=non_spec_state_indices_tensor,
                ).unsqueeze(0)
            else:
                core_attn_out_non_spec, last_recurrent_state = None, None
        else:
            # Qwen3.5: torch_npu ops do not support float32 ssm_state, use FLA ops instead.
            # NOTE: Once torch_npu supports float32 ssm_state, this branch can be removed.
            if attn_metadata.num_prefills > 0 or spec_sequence_masks is not None:
                g, beta = fused_gdn_gating_patch(self.A_log, a, b, self.dt_bias)
                if spec_sequence_masks is not None:
                    if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                        g_spec = g
                        beta_spec = beta
                        g_non_spec = None
                        beta_non_spec = None
                    else:
                        g_spec = g.index_select(1, spec_token_indx)
                        beta_spec = beta.index_select(1, spec_token_indx)
                        g_non_spec = g.index_select(1, non_spec_token_indx)
                        beta_non_spec = beta.index_select(1, non_spec_token_indx)
                else:
                    g_spec = None
                    beta_spec = None
                    g_non_spec = g
                    beta_non_spec = beta

                # 2.1: Process the multi-query part
                if spec_sequence_masks is not None:
                    core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                        q=query_spec,
                        k=key_spec,
                        v=value_spec,
                        g=g_spec,
                        beta=beta_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                        ssm_state_indices=spec_state_indices_tensor,
                        num_accepted_tokens=num_accepted_tokens,
                        use_qk_l2norm_in_kernel=True,
                    )
                else:
                    core_attn_out_spec, last_recurrent_state = None, None

                # 2.2: Process the remaining part
                if attn_metadata.num_prefills > 0:
                    if is_all_mode:
                        # All-mode: single-pass with intermediate state scatter
                        # (Qwen3.5: pool=[H,K,V]=kernel=[H,K,V], transpose_state=False)
                        initial_state = _build_initial_state(
                            ssm_state, attn_metadata,
                            attn_metadata.num_decodes, attn_metadata.num_prefills,
                            transpose_state=False,
                        )
                        non_spec_chunked_prefill_meta = getattr(
                            attn_metadata, "non_spec_chunked_prefill_meta", None
                        )
                        (core_attn_out_non_spec, last_recurrent_state, intermediate_states) = (
                            chunk_gated_delta_rule(
                                q=query_non_spec,
                                k=key_non_spec,
                                v=value_non_spec,
                                g=g_non_spec,
                                beta=beta_non_spec,
                                initial_state=initial_state,
                                output_final_state=True,
                                cu_seqlens=non_spec_query_start_loc,
                                prebuilt_meta=non_spec_chunked_prefill_meta,
                                head_first=False,
                                use_qk_l2norm_in_kernel=True,
                                return_intermediate_states=True,
                                state_dtype=ssm_state.dtype,
                            )
                        )
                        _write_final_states(ssm_state, last_recurrent_state,
                                            attn_metadata, transpose_state=False)
                        if intermediate_states is not None:
                            _scatter_intermediate_states(
                                ssm_state, intermediate_states, attn_metadata,
                                attn_metadata.num_decodes, transpose_state=False,
                            )
                    else:
                        initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
                        initial_state[~has_initial_state, ...] = 0
                        non_spec_chunked_prefill_meta = getattr(attn_metadata, "non_spec_chunked_prefill_meta", None)
                        (core_attn_out_non_spec, last_recurrent_state) = chunk_gated_delta_rule(
                            q=query_non_spec,
                            k=key_non_spec,
                            v=value_non_spec,
                            g=g_non_spec,
                            beta=beta_non_spec,
                            initial_state=initial_state,
                            output_final_state=True,
                            cu_seqlens=non_spec_query_start_loc,
                            prebuilt_meta=non_spec_chunked_prefill_meta,
                            head_first=False,
                            use_qk_l2norm_in_kernel=True,
                        )
                        ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)
                elif attn_metadata.num_decodes > 0:
                    core_attn_out_non_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_non_spec,
                        beta=beta_non_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                        ssm_state_indices=non_spec_state_indices_tensor,
                        use_qk_l2norm_in_kernel=True,
                    )
                else:
                    core_attn_out_non_spec, last_recurrent_state = None, None
            elif attn_metadata.num_decodes > 0:
                core_attn_out_spec = None
                core_attn_out_non_spec = fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log.contiguous(),
                    dt_bias=self.dt_bias.contiguous(),
                    q=query_non_spec.contiguous(),
                    k=key_non_spec.contiguous(),
                    v=value_non_spec.contiguous(),
                    a=a.contiguous(),
                    b=b.contiguous(),
                    initial_state_source=ssm_state,
                    initial_state_indices=non_spec_state_indices_tensor,
                    cu_seqlens=non_spec_query_start_loc,
                    use_qk_l2norm_in_kernel=True,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                )
            else:
                core_attn_out_spec, core_attn_out_non_spec = None, None
            maybe_save_kv_layer_to_connector("", [])

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)[:num_actual_tokens]
        elif spec_sequence_masks is not None:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)[:num_actual_tokens]
        else:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)[:num_actual_tokens]
