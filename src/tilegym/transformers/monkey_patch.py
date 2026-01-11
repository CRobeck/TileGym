# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import inspect

from transformers import PreTrainedModel

from tilegym import set_backend
from tilegym.logger import get_logger
from tilegym.ops.cutile.rope import get_apply_rope_func
from tilegym.ops.fused_swiglu import PartiallyFusedSwiGLUMLP
from tilegym.ops.cutile.rms_norm import TileRMSNorm
from tilegym.transformers.deepseek2.modeling_deepseek import DeepseekV2MoETileGym
from tilegym.transformers.deepseek2.modeling_deepseek import tilegym_deepseek_v2_forward

logger = get_logger(__name__)




def apply_tilegym_kernel_to_deepseek_v2(
    rope: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    attn: bool = True,
    moe: bool = True,
    model: PreTrainedModel = None,
    use_cutile: bool = False,
) -> None:
    """
    Apply TileGym kernels to replace original implementation in HuggingFace DeepSeek V2 models

    Args:
        rope (bool): Whether to apply TileGym's rotary position embedding. Default is True.
        rms_norm (bool): Whether to apply TileGym's RMSNorm. Default is True.
        swiglu (bool): Whether to apply TileGym's SwiGLU MLP for shared experts. Default is True.
        attn (bool): Whether to apply TileGym's Multi-head Latent Attention. Default is True.
        moe (bool): Whether to apply TileGym's fused MoE. Default is True.
        model (PreTrainedModel): The model instance to apply kernels to, if the model has already been
        loaded. Default is None.
        use_cutile (bool): Whether to use cutile backend. Default is False.
    """
    logger.info("--------------------------------")
    logger.info("apply_tilegym_kernel_to_deepseek_v2")
    logger.info("--------------------------------")
    from transformers.models.deepseek_v2 import modeling_deepseek_v2 as modeling_deepseek

    modeling_deepseek.apply_rotary_emb = get_apply_rope_func()

    modeling_deepseek.DeepseekV2RMSNorm = TileRMSNorm

     # Replace DeepseekV2MLP with TileGym's FUSED SwiGLU implementation
     # This eliminates ALL PyTorch linear operations by fusing gate+up+down projections.
     # This is critical for shared experts which run on every token.
    modeling_deepseek.DeepseekV2MLP = PartiallyFusedSwiGLUMLP


     # Replace attention forward with TileGym implementation
    modeling_deepseek.DeepseekV2Attention.forward = tilegym_deepseek_v2_forward

    modeling_deepseek.DeepseekV2MoE = DeepseekV2MoETileGym



def _apply_tilegym_kernel(model_type: str, **kwargs) -> None:
    """
    Applies TileGym kernels based on the specified model type. The custom
    kernels for the specified model type will be applied with the provided
    keyword arguments, otherwise the default configuration will be used.

    ** Note: This must be called before model initialization.

    Args:
        - model_type: the model types as defined in transformers/models/auto/modeling_auto.py
          and specified in the model's config.json
        - kwargs: keyword arguments that are passed to the corresponding apply_TileGym_kernel_to_* function.
    """
    apply_fn_signature = inspect.signature(apply_tilegym_kernel_to_deepseek_v2)

    # Filter out the keyword arguments that are not supported by the apply function
    applicable_kwargs = {key: value for key, value in kwargs.items() if key in apply_fn_signature.parameters}

    logger.info(f"Applying TileGym kernels for model type: {model_type} with kwargs: {applicable_kwargs}")

    # Assume this is invoked pre-model initialization, so we only need to patch transformers code
    apply_fn(**applicable_kwargs)
