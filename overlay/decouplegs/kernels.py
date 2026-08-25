from __future__ import annotations

import math

import torch
from torch import Tensor

from .transforms import real_sh_basis

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in minimal CPU environments
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _indexed_sh_kernel(
        directions,
        codebook,
        indices,
        output,
        count,
        degree: tl.constexpr,
        coefficients: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < count
        code = tl.load(indices + offsets, mask=mask, other=0).to(tl.int32)
        x = tl.load(directions + offsets * 3, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(directions + offsets * 3 + 1, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(directions + offsets * 3 + 2, mask=mask, other=0.0).to(tl.float32)
        if degree >= 1:
            inverse_norm = tl.rsqrt(x * x + y * y + z * z)
            x *= inverse_norm
            y *= inverse_norm
            z *= inverse_norm

        for channel in tl.static_range(0, 3):
            base = code * coefficients * 3 + channel
            result = 0.2820947917738781 * tl.load(codebook + base, mask=mask, other=0.0)
            if degree >= 1:
                result += 0.48860251190292 * (
                    -y * tl.load(codebook + base + 3, mask=mask, other=0.0)
                    + z * tl.load(codebook + base + 6, mask=mask, other=0.0)
                    - x * tl.load(codebook + base + 9, mask=mask, other=0.0)
                )
            if degree >= 2:
                z2 = z * z
                tmp0 = -1.092548430592079 * z
                c1 = x * x - y * y
                s1 = 2.0 * x * y
                result += (
                    0.5462742152960395 * s1 * tl.load(codebook + base + 12, mask=mask, other=0.0)
                    + tmp0 * y * tl.load(codebook + base + 15, mask=mask, other=0.0)
                    + (0.9461746957575601 * z2 - 0.3153915652525201)
                    * tl.load(codebook + base + 18, mask=mask, other=0.0)
                    + tmp0 * x * tl.load(codebook + base + 21, mask=mask, other=0.0)
                    + 0.5462742152960395 * c1 * tl.load(codebook + base + 24, mask=mask, other=0.0)
                )
            if degree >= 3:
                tmp0 = -2.285228997322329 * z2 + 0.4570457994644658
                tmp1 = 1.445305721320277 * z
                c2 = x * c1 - y * s1
                s2 = x * s1 + y * c1
                result += (
                    -0.5900435899266435 * s2 * tl.load(codebook + base + 27, mask=mask, other=0.0)
                    + tmp1 * s1 * tl.load(codebook + base + 30, mask=mask, other=0.0)
                    + tmp0 * y * tl.load(codebook + base + 33, mask=mask, other=0.0)
                    + z
                    * (1.865881662950577 * z2 - 1.119528997770346)
                    * tl.load(codebook + base + 36, mask=mask, other=0.0)
                    + tmp0 * x * tl.load(codebook + base + 39, mask=mask, other=0.0)
                    + tmp1 * c1 * tl.load(codebook + base + 42, mask=mask, other=0.0)
                    - 0.5900435899266435 * c2 * tl.load(codebook + base + 45, mask=mask, other=0.0)
                )
            tl.store(output + offsets * 3 + channel, result, mask=mask)


    @triton.jit
    def _merge_partition(static_ids, dynamic_ids, static_count, dynamic_count, diagonal):
        """Stable merge-path partition; static keys precede equal dynamic keys."""

        low = tl.maximum(0, diagonal - dynamic_count)
        high = tl.minimum(diagonal, static_count)
        for _ in tl.static_range(0, 32):
            active = low < high
            middle = (low + high) // 2
            dynamic_index = diagonal - middle
            static_key = tl.load(
                static_ids + middle,
                mask=active & (middle < static_count),
                other=0x7FFFFFFFFFFFFFFF,
            )
            dynamic_previous = tl.load(
                dynamic_ids + dynamic_index - 1,
                mask=active & (dynamic_index > 0),
                other=-1,
            )
            # Equality moves the partition right so the cached static stream
            # remains stable before the newly generated dynamic stream.
            move_right = active & (dynamic_index > 0) & (
                dynamic_previous >= static_key
            )
            low = tl.where(move_right, middle + 1, low)
            high = tl.where(active & ~move_right, middle, high)
        return low


    @triton.jit
    def _pair_sides(item, stage: tl.constexpr, dimensions: tl.constexpr):
        outer: tl.constexpr = item.numel >> dimensions
        shape: tl.constexpr = [outer * 2**stage, 2, 2 ** (dimensions - stage - 1)]
        pair = tl.arange(0, 2)[None, :, None]
        shaped = tl.reshape(item, shape)
        left = tl.broadcast_to(tl.sum(shaped * (1 - pair), axis=1)[:, None, :], shape)
        right = tl.broadcast_to(tl.sum(shaped * pair, axis=1)[:, None, :], shape)
        return tl.reshape(left, item.shape), tl.reshape(right, item.shape)


    @triton.jit
    def _pair_compare_and_swap(
        keys,
        stable_ranks,
        values,
        flip,
        stage: tl.constexpr,
        dimensions: tl.constexpr,
    ):
        left_key, right_key = _pair_sides(keys, stage, dimensions)
        left_rank, right_rank = _pair_sides(stable_ranks, stage, dimensions)
        left_value, right_value = _pair_sides(values, stage, dimensions)
        should_swap = (left_key > right_key) | (
            (left_key == right_key) & (left_rank > right_rank)
        )
        should_swap = should_swap ^ flip
        keys = keys ^ tl.where(
            should_swap,
            left_key ^ right_key,
            tl.zeros_like(keys),
        )
        stable_ranks = stable_ranks ^ tl.where(
            should_swap,
            left_rank ^ right_rank,
            tl.zeros_like(stable_ranks),
        )
        values = values ^ tl.where(
            should_swap,
            left_value ^ right_value,
            tl.zeros_like(values),
        )
        return keys, stable_ranks, values


    @triton.jit
    def _merge_sorted_index_streams_kernel(
        static_ids,
        static_values,
        dynamic_ids,
        dynamic_values,
        output,
        static_count,
        dynamic_count,
        dynamic_index_offset,
        block_size: tl.constexpr,
        dimensions: tl.constexpr,
    ):
        block = tl.program_id(0)
        diagonal_start = block * block_size
        total = static_count + dynamic_count
        diagonal_stop = tl.minimum(diagonal_start + block_size, total)
        static_start = _merge_partition(
            static_ids,
            dynamic_ids,
            static_count,
            dynamic_count,
            diagonal_start,
        )
        static_stop = _merge_partition(
            static_ids,
            dynamic_ids,
            static_count,
            dynamic_count,
            diagonal_stop,
        )
        dynamic_start = diagonal_start - static_start
        dynamic_stop = diagonal_stop - static_stop
        static_length = static_stop - static_start
        dynamic_length = dynamic_stop - dynamic_start

        offsets = tl.arange(0, block_size)
        is_static = offsets < static_length
        is_dynamic = offsets >= block_size - dynamic_length
        static_index = static_start + offsets
        # Place the ascending dynamic segment in reverse order at the right of
        # the block. Static asc + sentinel + dynamic desc is one bitonic run.
        dynamic_index = dynamic_start + (block_size - 1 - offsets)
        maximum = 0x7FFFFFFFFFFFFFFF
        keys = tl.where(
            is_static,
            tl.load(static_ids + static_index, mask=is_static, other=maximum),
            tl.where(
                is_dynamic,
                tl.load(dynamic_ids + dynamic_index, mask=is_dynamic, other=maximum),
                maximum,
            ),
        )
        static_rank = static_index.to(tl.int64)
        dynamic_rank = (static_count + dynamic_index).to(tl.int64)
        stable_ranks = tl.where(
            is_static,
            static_rank,
            tl.where(is_dynamic, dynamic_rank, maximum),
        )
        static_value = tl.load(static_values + static_index, mask=is_static, other=0)
        dynamic_value = tl.load(dynamic_values + dynamic_index, mask=is_dynamic, other=0)
        values = tl.where(
            is_static,
            static_value,
            tl.where(
                is_dynamic,
                dynamic_value + dynamic_index_offset,
                0,
            ),
        ).to(tl.int32)

        # The input is already bitonic, so only the final merge network is
        # required instead of a full O(log^2 N) block sort.
        for stage in tl.static_range(0, dimensions):
            keys, stable_ranks, values = _pair_compare_and_swap(
                keys,
                stable_ranks,
                values,
                0,
                stage,
                dimensions,
            )
        tl.store(
            output + diagonal_start + offsets,
            values,
            mask=offsets < diagonal_stop - diagonal_start,
        )


def indexed_spherical_harmonics(
    directions: Tensor,
    codebook: Tensor,
    indices: Tensor,
    degree: int,
) -> Tensor:
    """Evaluate SH while reading VQ coefficients directly through integer indices."""

    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape [N, 3]")
    coefficients = (degree + 1) ** 2
    if codebook.ndim == 2:
        if codebook.shape[1] != coefficients * 3:
            raise ValueError("flattened codebook does not match the requested SH degree")
        shaped_codebook = codebook.reshape(-1, coefficients, 3)
    elif codebook.ndim == 3 and codebook.shape[1:] == (coefficients, 3):
        shaped_codebook = codebook
    else:
        raise ValueError("codebook must have shape [K, B*3] or [K, B, 3]")
    if indices.shape != (directions.shape[0],):
        raise ValueError("indices must contain one entry per direction")
    if directions.device != codebook.device or directions.device != indices.device:
        raise ValueError("directions, codebook, and indices must share a device")
    if directions.device.type != "cuda" or triton is None:
        basis = real_sh_basis(directions, degree)
        selected = shaped_codebook[indices.to(torch.int64)]
        return torch.einsum("nb,nbc->nc", basis, selected)
    if directions.dtype != torch.float32 or codebook.dtype != torch.float32:
        raise ValueError("the indexed CUDA SH kernel currently requires float32")
    output = torch.empty_like(directions)
    block_size = 256
    grid = (triton.cdiv(directions.shape[0], block_size),)
    _indexed_sh_kernel[grid](
        directions.contiguous(),
        shaped_codebook.contiguous(),
        indices.contiguous(),
        output,
        count=directions.shape[0],
        degree=degree,
        coefficients=coefficients,
        block_size=block_size,
    )
    return output


def sorted_index_merge_backend(
    static_count: int,
    dynamic_count: int,
    device: torch.device,
    *,
    triton_min_dynamic_ratio: float = 0.5,
) -> str:
    if static_count <= 0 or dynamic_count <= 0:
        return "copy"
    if (
        device.type == "cuda"
        and triton is not None
        and dynamic_count >= static_count * triton_min_dynamic_ratio
    ):
        return "triton_merge_path"
    return "torch_dynamic_search"


def merge_sorted_index_streams(
    static_ids: Tensor,
    static_values: Tensor,
    dynamic_ids: Tensor,
    dynamic_values: Tensor,
    *,
    dynamic_index_offset: int,
    block_size: int = 256,
    triton_min_dynamic_ratio: float = 0.5,
) -> Tensor:
    """Stable merge of two sorted intersection streams on CUDA.

    The Triton path emits only the flattened primitive ids required by gsplat;
    callers can combine per-tile counts directly and avoid allocating merged
    int64 keys. CPU and minimal environments retain an exact PyTorch fallback.
    """

    if static_ids.ndim != 1 or dynamic_ids.ndim != 1:
        raise ValueError("sorted ids must be one-dimensional")
    if static_values.shape != static_ids.shape or dynamic_values.shape != dynamic_ids.shape:
        raise ValueError("each sorted id needs one value")
    if static_ids.dtype != torch.int64 or dynamic_ids.dtype != torch.int64:
        raise ValueError("sorted ids must use int64")
    if static_values.dtype != torch.int32 or dynamic_values.dtype != torch.int32:
        raise ValueError("sorted values must use int32")
    tensors = (static_ids, static_values, dynamic_ids, dynamic_values)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all merge tensors must share a device")
    if dynamic_index_offset < 0:
        raise ValueError("dynamic_index_offset must be non-negative")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if triton_min_dynamic_ratio < 0:
        raise ValueError("triton_min_dynamic_ratio must be non-negative")

    static_count = static_ids.numel()
    dynamic_count = dynamic_ids.numel()
    if dynamic_count == 0:
        return static_values
    if static_count == 0:
        return dynamic_values + dynamic_index_offset
    output = torch.empty(
        static_count + dynamic_count,
        dtype=static_values.dtype,
        device=static_values.device,
    )
    if (
        sorted_index_merge_backend(
            static_count,
            dynamic_count,
            static_ids.device,
            triton_min_dynamic_ratio=triton_min_dynamic_ratio,
        )
        == "triton_merge_path"
    ):
        _merge_sorted_index_streams_kernel[(triton.cdiv(output.numel(), block_size),)](
            static_ids.contiguous(),
            static_values.contiguous(),
            dynamic_ids.contiguous(),
            dynamic_values.contiguous(),
            output,
            static_count,
            dynamic_count,
            dynamic_index_offset,
            block_size=block_size,
            dimensions=int(math.log2(block_size)),
        )
        return output

    # Search only the smaller dynamic stream. This multi-kernel path is faster
    # than blockwise merge-path when dynamic intersections are sparse.
    dynamic_positions = torch.searchsorted(
        static_ids,
        dynamic_ids,
        right=True,
    ) + torch.arange(dynamic_count, device=dynamic_ids.device)
    static_positions = torch.ones(
        output.numel(), dtype=torch.bool, device=static_values.device
    )
    output[dynamic_positions] = dynamic_values + dynamic_index_offset
    static_positions[dynamic_positions] = False
    output[static_positions] = static_values
    return output
