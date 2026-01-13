# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import argparse
import datetime
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.profiler import record_function
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


#from tilegym.transformers import apply_tilegym_kernel_to_deepseek_v2
# from tilegym.transformers import apply_tilegym_kernel_to_llama
from transformers.models.deepseek_v2 import modeling_deepseek_v2 as modeling_deepseek
from tilegym.logger import get_logger
from tilegym.ops.cutile.rope import get_apply_rope_func
from tilegym.ops.fused_swiglu import PartiallyFusedSwiGLUMLP
from tilegym.ops.cutile.rms_norm import TileRMSNorm
from tilegym.transformers.deepseek2.modeling_deepseek import DeepseekV2MoETileGym
from tilegym.transformers.deepseek2.modeling_deepseek import tilegym_deepseek_v2_forward

logger = get_logger(__name__)


def check_and_setup_model_cache(model_id):
    """
    Check if model exists in local cache, set up cache directories if needed.
    Returns the effective cache directory for the model.
    """
    # Get cache directory from environment or use default
    cache_base = os.environ.get("TILEGYM_MODEL_CACHE_DIR", os.path.expanduser("~/.cache"))

    # Set up HuggingFace cache directories
    hf_home = os.path.join(cache_base, "huggingface")
    hf_hub = os.path.join(hf_home, "hub")

    # Set environment variables for transformers
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", hf_hub)

    # Create cache directories if they don't exist
    Path(hf_hub).mkdir(parents=True, exist_ok=True)

    return str(cache_base)


def _check_cache_complete(model_id):
    """Check if cache directory exists with snapshots. Actual completeness verified during load."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hf_hub = os.path.join(hf_home, "hub")
    model_cache_path = Path(hf_hub) / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = model_cache_path / "snapshots"

    # Check if snapshots directory exists and has content
    return snapshots_dir.exists() and any(snapshots_dir.iterdir())


def _load_with_fallback(model_id, loader_class, resource_name, **kwargs):
    """
    Generic loader with automatic fallback from local cache to download.
    Supports both HuggingFace model IDs and local filesystem paths.
    """
    # Check if model_id is a local path
    model_path = Path(model_id)
    if model_path.exists() and model_path.is_dir():
        print(f"Loading {resource_name} from local path: {model_id}")
        try:
            result = loader_class.from_pretrained(model_id, **kwargs)
            print(f"Successfully loaded {resource_name} from local path")
            return result
        except Exception as e:
            print(f"Error loading {resource_name} from local path: {e}")
            raise

    # Otherwise, use HuggingFace Hub logic
    check_and_setup_model_cache(model_id)
    print(f"Loading {resource_name} {model_id}...")

    # Try local cache first if available
    if _check_cache_complete(model_id):
        print("Found cached files, attempting to use local cache")
        try:
            kwargs_local = kwargs.copy()
            kwargs_local["local_files_only"] = True
            result = loader_class.from_pretrained(model_id, **kwargs_local)
            print(f"Successfully loaded {resource_name} from local cache")
            return result
        except Exception:
            print("Failed to load from local cache, will download from HuggingFace")

    # Fallback to download
    try:
        result = loader_class.from_pretrained(model_id, **kwargs)
        print(f"Successfully loaded {resource_name}")
        return result
    except Exception as e:
        print(f"Error loading {resource_name}: {e}")
        raise


def load_model_with_cache(model_id, **kwargs):
    """Load model with cache checking. Downloads if not available locally."""
    return _load_with_fallback(model_id, AutoModelForCausalLM, "model", **kwargs)


def load_tokenizer_with_cache(model_id):
    """Load tokenizer with cache checking."""
    return _load_with_fallback(model_id, AutoTokenizer, "tokenizer")


class NaiveForwardWrapper:
    def __init__(
        self,
        model,
        tokenizer,
        messages_list,
        args,
        device="cuda",
        **default_kwargs,
    ):
        self.model = model
        self.default_kwargs = default_kwargs
        self.tokenizer = tokenizer
        self.tokenized_inputs = tokenizer(messages_list, return_tensors="pt").to(device)
        self.input_seq_len = self.tokenized_inputs.input_ids.shape[1]

        self.default_kwargs.update(
            {
                "input_ids": self.tokenized_inputs.input_ids,
                "attention_mask": self.tokenized_inputs.attention_mask,
                "max_new_tokens": args.output_length,
                "min_new_tokens": args.output_length,  # Force to generate exactly output_length tokens
            }
        )

    def get_input_seq_len(self):
        return self.input_seq_len

    def update_params(self, **kwargs):
        self.default_kwargs.update(kwargs)

    def forward(self):
        return self.model.generate(**self.default_kwargs)

    def post_process(self, outputs):
        return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference")
    parser.add_argument("--use_tilegym", action="store_true", help="Use tilegym kernel")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Model ID to load")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument(
        "--input_text",
        type=str,
        default="What is the capital of France?",
        help="Input text for generation",
    )
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs for averaging performance")
    parser.add_argument("--warmup_runs", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--show_outputs", action="store_true", help="Show full model outputs")
    parser.add_argument("--summary_file", type=str, default=None, help="File to append summary lines")
    # use-attn: True or False
    parser.add_argument("--use_attn", action="store_true", help="Use attention")
    # use-cutile: True or False
    parser.add_argument("--use_cutile", action="store_true", help="Use cutile")
    # profile: True or False
    parser.add_argument("--profile", action="store_true", help="Profile the model")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory to save profiler logs (default: /logs)",
    )
    # precision: bfloat16 or float32
    parser.add_argument("--precision", type=str, default="bfloat16", help="Precision")
    parser.add_argument("--sentence_file", type=str, default=None)
    # output length
    parser.add_argument("--output_length", type=int, default=100, help="Output length")
    # Only used in benchmark
    # If mock_input_len > 0, then the input length will be set to mock_input_len, this is used to mock the input length
    # If you use this, you may not get the correct answer of your sentence
    parser.add_argument("--mock_input_len", type=int, default=0, help="Mock input length")
    return parser.parse_args()


def get_messages_list(args):
    messages_list = []
    if args.mock_input_len > 0:
        line = " Hello" * (args.mock_input_len - 1)  # Because `<|begin_of_text|>` token will be added by the tokenizer
        for _ in range(args.batch_size):
            messages_list.append(line)
    elif args.sentence_file is not None:
        with open(args.sentence_file, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            for _ in range(args.batch_size):
                messages_list.append("\n".join(lines))
    else:
        for _ in range(args.batch_size):
            messages_list.append(args.input_text)
        print(messages_list)
    return messages_list


# def apply_tilegym_patch(model_id, use_attn=False, use_cutile=False):
#     model_name = model_id.lower()
#     # if "llama" in model_name:
#     #     apply_tilegym_kernel_to_llama(rope=True, swiglu=True, rms_norm=True, attn=use_attn, use_cutile=use_cutile)
#     elif "deepseek" in model_name:
#         apply_tilegym_kernel_to_deepseek_v2(
#             rope=True, rms_norm=True, swiglu=True, attn=use_attn, moe=True, use_cutile=use_cutile
#         )
#     else:
#         print(f"Warning: Model {model_id} is not supported in tilegym patch. No optimizations will be applied.")


class KernelFilter:
    def __init__(self):
        self.kernel_names_prefix = [
            # Rope kernels
            "rope",
            "tile_rope_kernel",
            # Attention kernels
            "prefill_fmha",
            "prefill_mla",
            "fmha_kernel",
            "attention_decode_kernel",
            # MoE kernels
            "moe",
            "fused_moe_kernel",
            "moe_align_block_size",
            # SwiGLU/activation kernels
            "silu_and_mul",
            "_silu_and_mul_kernel",
            "swiglu_forward",
            # General cutile prefixes
            "tile_",
            "cutile",
            # RMS norm kernels
            "rms_norm_kernel",
            # MLA kernels
            "naive_absorb_mla",
            "_mla_decoding",
            # Reduce kernels
            "splitk_reduce_kernel",
            # GEMM kernels
            "static_persistent_gemm_kernel",
            "static_persistent_matmul_kernel",
            "group_gemm",
            "matmul",
            "gemm_kernel",
        ]
        self.blacklist_kernel_names = [
            "aten::matmul",
        ]

    def get_kernel_names(self):
        return self.kernel_names_prefix

    def contains(self, key):
        for k in self.kernel_names_prefix:
            if k in key and key not in self.blacklist_kernel_names:
                return True
        return False


def main():
    args = parse_args()

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(
        f"Benchmark settings: model={args.model_id}, batch_size={args.batch_size}, output_length={args.output_length}"
    )

    # Load tokenizer and model with cache support
    print(f"Loading model {args.model_id}...")
    tokenizer = load_tokenizer_with_cache(args.model_id)
    # backend = "base"
    # if args.use_tilegym:
        # if args.use_cutile:
    backend = "cutile"
    print("########################")
    print("#######Use TileGym#########")
    print("########################")
    # apply_tilegym_patch(args.model_id, args.use_attn, use_cutile=True)
    # apply_tilegym_kernel_to_deepseek_v2(
    #     rope=True, rms_norm=True, swiglu=True, attn=True, moe=True, use_cutile=True
    # )
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
    modeling_deepseek.DeepseekV2Attention.forward = tilegym_deepseek_v2_forward

    modeling_deepseek.DeepseekV2MoE = DeepseekV2MoETileGym

    # Load model with cache support
    model_kwargs = {
        "trust_remote_code": False,
        "device_map": "cuda",
        "torch_dtype": torch.bfloat16 if args.precision == "bfloat16" else torch.float32,
    }

    model = load_model_with_cache(args.model_id, **model_kwargs)

    if args.show_outputs or args.profile:
        args.warmup_runs = 1
        if args.show_outputs:
            args.num_runs = 1
        do_sample = False
    else:
        do_sample = True

    forward_wrapper = NaiveForwardWrapper(
        model,
        tokenizer=tokenizer,
        messages_list=get_messages_list(args),
        args=args,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        early_stopping=False,
    )

    # Warmup runs
    print("Performing warmup runs...")
    for i in range(args.warmup_runs):
        with torch.no_grad():
            _ = forward_wrapper.forward()
        print(f"  Warmup run {i + 1}/{args.warmup_runs} completed")

    print(f"\nRunning benchmark with {args.num_runs} iterations...")
    generation_times = []
    tokens_per_second = []
    torch.cuda.synchronize()
    for run in range(args.num_runs):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        total_tokens = 0
        outputs_list = []
        input_length = forward_wrapper.get_input_seq_len()

        start_event.record()
        with torch.no_grad():
            outputs = forward_wrapper.forward()
        end_event.record()

        outputs = forward_wrapper.post_process(outputs)
        outputs_list.append(outputs)
        # Calculate only newly generated tokens
        generated_tokens = outputs.shape[1] - input_length
        print(f"generated_tokens: {generated_tokens}")
        total_tokens += generated_tokens * args.batch_size

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
        generation_times.append(elapsed_time)
        tokens_per_second.append(total_tokens / elapsed_time)

        print(
            f"Run {run + 1}: Generated {generated_tokens} tokens in {elapsed_time:.4f}s ({generated_tokens / elapsed_time:.2f} tokens/sec)"
        )

    # Print results
    avg_time = np.mean(generation_times)
    avg_tokens_per_sec = np.mean(tokens_per_second)
    std_tokens_per_sec = np.std(tokens_per_second)

    print("\n===== BENCHMARK RESULTS =====")
    print(f"Model: {args.model_id}")
    print(f"Device: {device}")
    print(f"Use TileGym: {args.use_tilegym}")
    print(f"Backend: {backend}")
    print(f"Use attention: {args.use_attn}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output length: {args.output_length}")
    print(f"Input length: {input_length}")
    print(f"Average generation time: {avg_time:.4f}s")
    print(f"Average throughput: {avg_tokens_per_sec:.2f} ± {std_tokens_per_sec:.2f} tokens/sec")

    # Display generated outputs if needed
    # if args.show_outputs:
    print("\n===== GENERATED OUTPUTS =====")
    for batch_idx, outputs in enumerate(outputs_list):
        for i in range(outputs.shape[0]):
            decoded_output = tokenizer.decode(outputs[i], skip_special_tokens=False)
            print(f"\nBatch {batch_idx + 1}, Output {i + 1}:")
            print(decoded_output)
            print("-" * 50)

    # Generate case identifier
    case_id = f"{args.model_id.split('/')[-1]}"

    # Write summary to file if specified
    if args.summary_file:
        if args.use_tilegym:
            if args.use_cutile:
                case_id += "_cutile"
            if args.use_attn:
                case_id += "_attn"
        else:
            case_id += "_naive"
        case_id += f"_{args.precision}"

        summary_line = f"{case_id:<40} | {avg_tokens_per_sec:>10.2f} | {avg_time:>9.4f}"
        # Note: CUDA kernel time will be added after profiling if --profile is used
        # For now, write the summary line without CUDA kernel time
        with open(args.summary_file, "a") as f:
            f.write(summary_line + "\n")
        print(f"Summary written to {args.summary_file}: {summary_line}")


if __name__ == "__main__":
    main()
