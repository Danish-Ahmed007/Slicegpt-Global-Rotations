# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Phase 1: Inference Throughput Measurement
# Measures tokens/sec, parameter savings, and memory for per-layer vs global PCA

import argparse
import csv
import gc
import logging
import os
import pathlib
import time
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.config import config
from slicegpt.slicing_scheduler import ConstSlicingScheduler


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure inference throughput for per-layer vs global PCA")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to benchmark")
    parser.add_argument("--sparsity", type=float, default=0.25, help="Sparsity level")
    parser.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--cal-nsamples", type=int, default=128, help="Calibration samples")
    parser.add_argument("--cal-batch-size", type=int, default=16, help="Calibration batch size")
    parser.add_argument("--cal-max-seqlen", type=int, default=2048, help="Calibration sequence length")
    parser.add_argument("--benchmark-seqlen", type=int, default=128, help="Sequence length for benchmarking")
    parser.add_argument("--benchmark-batch-size", type=int, default=1, help="Batch size for benchmarking")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of benchmark runs")
    parser.add_argument("--warmup-runs", type=int, default=10, help="Warmup runs before timing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/throughput_study")
    parser.add_argument("--hf-token", type=str, default=os.getenv('HF_TOKEN', None))
    parser.add_argument("--skip-perlayer", action="store_true", help="Skip per-layer benchmark (if already done)")
    parser.add_argument("--skip-global", action="store_true", help="Skip global benchmark (if already done)")
    return parser.parse_args()


def count_shortcut_parameters(model_adapter) -> dict:
    """Count parameters in shortcut matrices."""
    total_shortcut_params = 0
    attn_shortcut_params = 0
    mlp_shortcut_params = 0
    
    for layer_adapter in model_adapter.get_layers():
        layer = layer_adapter.layer
        if hasattr(layer, 'attn_shortcut_Q') and layer.attn_shortcut_Q is not None:
            attn_shortcut_params += layer.attn_shortcut_Q.numel()
        if hasattr(layer, 'mlp_shortcut_Q') and layer.mlp_shortcut_Q is not None:
            mlp_shortcut_params += layer.mlp_shortcut_Q.numel()
    
    total_shortcut_params = attn_shortcut_params + mlp_shortcut_params
    
    return {
        "total_shortcut_params": total_shortcut_params,
        "attn_shortcut_params": attn_shortcut_params,
        "mlp_shortcut_params": mlp_shortcut_params,
        "shortcut_memory_mb": total_shortcut_params * 2 / (1024 * 1024),  # fp16 = 2 bytes
    }


def count_total_parameters(model) -> int:
    """Count total model parameters."""
    return sum(p.numel() for p in model.parameters())


def sync_cuda():
    """Synchronize CUDA for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_generation(model, tokenizer, seq_len: int, batch_size: int, 
                         num_runs: int, warmup_runs: int) -> dict:
    """
    Benchmark token generation throughput.
    Measures time to generate tokens one-by-one (autoregressive).
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    input_ids = torch.randint(
        low=1, high=tokenizer.vocab_size - 1,
        size=(batch_size, seq_len),
        device=device
    )
    attention_mask = torch.ones_like(input_ids)
    
    # Warmup
    logging.info(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
    sync_cuda()
    
    # Benchmark: Prefill (process full sequence)
    logging.info(f"Benchmarking prefill (seq_len={seq_len})...")
    prefill_times = []
    for _ in tqdm(range(num_runs), desc="Prefill"):
        sync_cuda()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
        sync_cuda()
        prefill_times.append(time.perf_counter() - start)
        del outputs
    
    prefill_median = np.median(prefill_times)
    prefill_std = np.std(prefill_times)
    prefill_tokens_per_sec = (batch_size * seq_len) / prefill_median
    
    # Benchmark: Single token generation (decode step)
    logging.info("Benchmarking decode (single token generation)...")
    
    # First get KV cache from prefill
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
        past_kv = outputs.past_key_values
    
    # Generate one token at a time
    next_token = torch.randint(1, tokenizer.vocab_size - 1, (batch_size, 1), device=device)
    extended_mask = torch.ones((batch_size, seq_len + 1), device=device)
    
    decode_times = []
    for _ in tqdm(range(num_runs), desc="Decode"):
        sync_cuda()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(next_token, attention_mask=extended_mask, past_key_values=past_kv, use_cache=True)
        sync_cuda()
        decode_times.append(time.perf_counter() - start)
    
    decode_median = np.median(decode_times)
    decode_std = np.std(decode_times)
    decode_tokens_per_sec = batch_size / decode_median  # 1 token per batch item
    
    return {
        "prefill_median_ms": prefill_median * 1000,
        "prefill_std_ms": prefill_std * 1000,
        "prefill_tokens_per_sec": prefill_tokens_per_sec,
        "decode_median_ms": decode_median * 1000,
        "decode_std_ms": decode_std * 1000,
        "decode_tokens_per_sec": decode_tokens_per_sec,
        "seq_len": seq_len,
        "batch_size": batch_size,
    }


def prepare_and_slice_model(args, use_global_pca: bool):
    """Load, fuse, rotate and slice a model."""
    pca_type = "global" if use_global_pca else "per-layer"
    logging.info(f"\n{'='*60}")
    logging.info(f"Preparing model with {pca_type} PCA, sparsity={args.sparsity}")
    logging.info(f"{'='*60}")
    
    # Load fresh model
    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
        args.model, token=args.hf_token, dtype=config.dtype
    )
    model = model_adapter.model
    
    # Get original param count
    original_params = count_total_parameters(model)
    logging.info(f"Original parameters: {original_params:,}")
    
    # Load calibration data
    dataset = data_utils.get_dataset("wikitext2")
    train_loader = data_utils.prepare_dataloader(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=args.cal_max_seqlen,
        batch_size=args.cal_batch_size,
        nsamples=args.cal_nsamples,
        seed=args.seed,
    )
    
    # Replace and fuse layers
    layernorm_fusion.replace_layers(model_adapter)
    layernorm_fusion.fuse_modules(model_adapter)
    
    # Compute new dimension
    new_dim = int((1 - args.sparsity) * model_adapter.hidden_size)
    new_dim -= new_dim % 8  # round to multiple of 8
    logging.info(f"New embedding dimension: {new_dim}")
    
    scheduler = ConstSlicingScheduler(new_dim)
    
    # Time the compression
    sync_cuda()
    compress_start = time.perf_counter()
    
    if use_global_pca:
        rotate.rotate_and_slice_global_pca(
            model_adapter, train_loader, scheduler,
            apply_mask=True, final_orientation='pca'
        )
    else:
        rotate.rotate_and_slice(
            model_adapter, train_loader, scheduler,
            final_orientation='pca'
        )
    
    sync_cuda()
    compress_time = time.perf_counter() - compress_start
    logging.info(f"Compression time: {compress_time:.2f}s")
    
    # Count parameters
    sliced_params = count_total_parameters(model)
    shortcut_info = count_shortcut_parameters(model_adapter)
    
    logging.info(f"Sliced parameters: {sliced_params:,}")
    logging.info(f"Shortcut parameters: {shortcut_info['total_shortcut_params']:,}")
    logging.info(f"Shortcut memory: {shortcut_info['shortcut_memory_mb']:.2f} MB")
    
    return model_adapter, tokenizer, {
        "original_params": original_params,
        "sliced_params": sliced_params,
        "compress_time_s": compress_time,
        **shortcut_info,
    }


def main():
    utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    args = arg_parser()
    
    if args.device:
        config.device = torch.device(args.device)
    
    if args.dtype == "fp16":
        config.dtype = torch.float16
    else:
        config.dtype = torch.float32
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = pathlib.Path(args.model).name
    
    results = []
    
    # Benchmark per-layer PCA
    if not args.skip_perlayer:
        logging.info("\n" + "="*70)
        logging.info("BENCHMARKING: PER-LAYER PCA")
        logging.info("="*70)
        
        model_adapter, tokenizer, compress_info = prepare_and_slice_model(args, use_global_pca=False)
        model_adapter.model.to(config.device)
        model_adapter.use_cache = True
        
        bench_results = benchmark_generation(
            model_adapter.model, tokenizer,
            seq_len=args.benchmark_seqlen,
            batch_size=args.benchmark_batch_size,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs
        )
        
        results.append({
            "model": args.model,
            "sparsity": args.sparsity,
            "pca_type": "per-layer",
            **compress_info,
            **bench_results,
        })
        
        # Cleanup
        del model_adapter
        gc.collect()
        torch.cuda.empty_cache()
    
    # Benchmark global PCA
    if not args.skip_global:
        logging.info("\n" + "="*70)
        logging.info("BENCHMARKING: GLOBAL PCA")
        logging.info("="*70)
        
        model_adapter, tokenizer, compress_info = prepare_and_slice_model(args, use_global_pca=True)
        model_adapter.model.to(config.device)
        model_adapter.use_cache = True
        
        bench_results = benchmark_generation(
            model_adapter.model, tokenizer,
            seq_len=args.benchmark_seqlen,
            batch_size=args.benchmark_batch_size,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs
        )
        
        results.append({
            "model": args.model,
            "sparsity": args.sparsity,
            "pca_type": "global",
            **compress_info,
            **bench_results,
        })
        
        # Cleanup
        del model_adapter
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save results to CSV
    csv_path = output_dir / f"throughput_{model_name}_{args.sparsity}_{timestamp}.csv"
    if results:
        fieldnames = results[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logging.info(f"\nResults saved to: {csv_path}")
    
    # Print summary
    if len(results) == 2:
        perlayer = results[0]
        global_pca = results[1]
        
        logging.info("\n" + "="*70)
        logging.info("THROUGHPUT COMPARISON SUMMARY")
        logging.info("="*70)
        logging.info(f"Model: {args.model}")
        logging.info(f"Sparsity: {args.sparsity}")
        logging.info(f"Benchmark seq_len: {args.benchmark_seqlen}")
        logging.info("")
        
        logging.info("PREFILL (tokens/sec):")
        logging.info(f"  Per-layer: {perlayer['prefill_tokens_per_sec']:.1f}")
        logging.info(f"  Global:    {global_pca['prefill_tokens_per_sec']:.1f}")
        logging.info(f"  Speedup:   {global_pca['prefill_tokens_per_sec']/perlayer['prefill_tokens_per_sec']:.2f}x")
        
        logging.info("")
        logging.info("DECODE (tokens/sec):")
        logging.info(f"  Per-layer: {perlayer['decode_tokens_per_sec']:.1f}")
        logging.info(f"  Global:    {global_pca['decode_tokens_per_sec']:.1f}")
        logging.info(f"  Speedup:   {global_pca['decode_tokens_per_sec']/perlayer['decode_tokens_per_sec']:.2f}x")
        
        logging.info("")
        logging.info("SHORTCUT PARAMETERS SAVED:")
        shortcut_saved = perlayer['total_shortcut_params'] - global_pca['total_shortcut_params']
        memory_saved = perlayer['shortcut_memory_mb'] - global_pca['shortcut_memory_mb']
        logging.info(f"  Parameters: {shortcut_saved:,}")
        logging.info(f"  Memory:     {memory_saved:.2f} MB")
        
        logging.info("")
        logging.info("COMPRESSION TIME:")
        logging.info(f"  Per-layer: {perlayer['compress_time_s']:.1f}s")
        logging.info(f"  Global:    {global_pca['compress_time_s']:.1f}s")
        logging.info(f"  Speedup:   {perlayer['compress_time_s']/global_pca['compress_time_s']:.2f}x")


if __name__ == "__main__":
    main()
