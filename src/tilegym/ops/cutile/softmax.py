# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
import numpy as np
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

# Type aliases for constants
ConstInt = ct.Constant[int]


@ct.kernel(occupancy=4)
def softmax_kernel(
    output,
    input,
    n_rows: ConstInt,
    TILE_SIZE: ConstInt,
    DIM_COLS: ConstInt,
):
    # Static persistent scheduling: each block processes multiple rows
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)

    for row_idx in range(pid, n_rows, num_programs):
        # Load the row tile using index-based access
        # Note: TILE_SIZE is expected to be >= DIM_COLS (ensured by launch function)
        row = ct.load(input, index=(row_idx, 0), shape=(1, TILE_SIZE))

        # Create mask for valid columns
        col_offsets = ct.arange(TILE_SIZE, dtype=torch.int32)
        mask_col = col_offsets < DIM_COLS
        mask = mask_col[None, :]

        # Apply mask by setting invalid positions to -inf for numerical stability
        neg_inf_tile = ct.full((1, TILE_SIZE), -np.inf, dtype=row.dtype)
        row = ct.where(mask, row, neg_inf_tile)

        # Convert to float32 for computation
        row = ct.astype(row, torch.float32)

        # Subtract maximum for numerical stability
        row_max = ct.max(row, 1, keepdims=True)
        row_minus_max = ct.sub(row, row_max)

        # Compute exponential
        numerator = ct.exp(row_minus_max)

        # Compute sum for normalization
        denominator = ct.sum(numerator, 1, keepdims=True)

        # Final softmax computation
        softmax_output = ct.truediv(numerator, denominator)

        # Convert back to original dtype
        softmax_output = ct.astype(softmax_output, input.dtype)

        # Store result using index-based access
        ct.store(output, index=(row_idx, 0), tile=softmax_output)

# TMA version with static persistent scheduling
@ct.kernel(occupancy=2)
def softmax_kernel_tma(
    output,
    input,
    n_rows: ConstInt,
    n_cols: ConstInt,
    TILE_SIZE: ConstInt,
):
    # Static persistent scheduling: each block processes multiple rows
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)

    for row_idx in range(pid, n_rows, num_programs):
        # Load the entire row in one tile (TILE_SIZE >= n_cols by design)
        row = ct.load(input, index=(row_idx, 0), shape=(1, TILE_SIZE))

        # Create mask for valid columns
        col_offsets = ct.arange(TILE_SIZE, dtype=torch.int32)
        mask_col = col_offsets < n_cols
        mask = mask_col[None, :]

        # Apply mask by setting invalid positions to -inf
        neg_inf_tile = ct.full((1, TILE_SIZE), -np.inf, dtype=row.dtype)
        row = ct.where(mask, row, neg_inf_tile)

        # Convert to float32 for computation
        row = ct.astype(row, np.float32)

        # Subtract maximum for numerical stability
        row_max = ct.max(row, 1, keepdims=True)
        row_minus_max = ct.sub(row, row_max)

        # Compute exponential
        numerator = ct.exp(row_minus_max)

        # Compute sum for normalization
        denominator = ct.sum(numerator, 1, keepdims=True)

        # Final softmax computation
        softmax_output = ct.truediv(numerator, denominator)

        # Convert back to original dtype and store
        softmax_output = ct.astype(softmax_output, input.dtype)
        ct.store(output, index=(row_idx, 0), tile=softmax_output)



# Launch patterns for the kernels:
def launch_softmax_kernel(input, output, TILE_SIZE=1024):
    """
    Launch the basic cuTile softmax kernel with static persistent scheduling

    Args:
        input: Input tensor of shape (n_rows, n_cols)
        output: Output tensor of shape (n_rows, n_cols)
        TILE_SIZE: Tile size for processing
    """
    n_rows, n_cols = input.shape
    original_n_cols = n_cols

    # Pad tensors if needed to match TILE_SIZE
    # This is required because ct.load expects shape to match tensor dimensions
    needs_padding = n_cols < TILE_SIZE
    if needs_padding:
        padding = TILE_SIZE - n_cols
        input_padded = torch.nn.functional.pad(input, (0, padding), value=-float('inf'))
        output_padded = torch.nn.functional.pad(output, (0, padding), value=0.0)
    else:
        input_padded = input
        output_padded = output

    # Ensure tensors are contiguous
    input_padded = input_padded.contiguous()
    output_padded = output_padded.contiguous()

    NUM_SM = torch.cuda.get_device_properties(input.device).multi_processor_count
    occupancy = 4  # Match @ct.kernel(occupancy=4)
    num_programs = min(NUM_SM * occupancy, n_rows)
    grid = (num_programs, 1, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        softmax_kernel,
        (
            output_padded,
            input_padded,
            n_rows,
            TILE_SIZE,
            original_n_cols,
        ),
    )

    # Copy result back to original output tensor if padding was used
    if needs_padding:
        output.copy_(output_padded[:, :original_n_cols])


def launch_softmax_kernel_tma(
    input,
    output,
):
    """
    Launch the TMA cuTile softmax kernel

    Args:
        input: Input tensor of shape (n_rows, n_cols)
        output: Output tensor of shape (n_rows, n_cols)
    """
    # Ensure input is 2D
    original_shape = input.shape
    if input.dim() == 1:
        input = input.unsqueeze(0)
        output = output.unsqueeze(0)
    elif input.dim() > 2:
        input = input.view(-1, input.shape[-1])
        output = output.view(-1, output.shape[-1])

    n_rows, n_cols = input.shape

    TILE_SIZE = next_power_of_2(n_cols)
    original_n_cols = n_cols

    # Regular TMA path (single tile per row, persistent scheduling)
    softmax_kernel_forward = softmax_kernel_tma

    # Pad if needed
    if n_cols < TILE_SIZE:
        padding = TILE_SIZE - n_cols
        input = torch.nn.functional.pad(input, (0, padding), value=-float('inf'))
        output = torch.nn.functional.pad(output, (0, padding), value=0.0)

    # Ensure tensors are contiguous
    input = input.contiguous()
    output = output.contiguous()

    NUM_SM = torch.cuda.get_device_properties(input.device).multi_processor_count
    occupancy = 2  # Match @ct.kernel(occupancy=2)
    num_programs = min(NUM_SM * occupancy, n_rows)
    grid = (num_programs, 1, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        softmax_kernel_forward,
        (
            output,
            input,
            n_rows,
            original_n_cols,
            TILE_SIZE,
        ),
    )

    # Trim output to original size
    if original_n_cols < output.shape[-1]:
        output = output[..., :original_n_cols]


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        use_tma=False,
        ):
        n_rows, n_cols = x.shape

        # Create output tensor
        y = torch.empty_like(x)

        if use_tma:
            # Use TMA implementation
            launch_softmax_kernel_tma(
                x,
                y,
            )
        else:
            # Use grid-based implementation - ensure single tile per row for correctness
            TILE_SIZE = next_power_of_2(n_cols)
            launch_softmax_kernel(x, y, TILE_SIZE=TILE_SIZE)
        return y

@register_impl("softmax", backend="cutile")
def softmax(
    x,
    use_tma=False,
    **kwargs,
):
    """
    Performs softmax using cuTile kernels with automatic gradient support

    Args:
        x: Input tensor of shape (M, N)
        use_tma: Whether to use TMA (Tensor Memory Accelerator) implementation.
                Requires H100+ GPU (compute capability >= 9.0)
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Softmax output tensor with gradient support
    """
    return Softmax.apply(
        x,
        use_tma,
    )
