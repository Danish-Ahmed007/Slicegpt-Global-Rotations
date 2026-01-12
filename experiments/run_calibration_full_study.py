#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Phase 3: FULL Calibration Token Count Ablation Study
# 
# This script runs a comprehensive ablation across:
# - Models: OPT-125M, OPT-1.3B, OPT-2.7B, OPT-6.7B
# - Sparsities: 10%, 20%, 25%, 30%
# - Calibration samples: 16, 32, 64, 128, 256
# - PCA types: per-layer, global
#
# Total experiments: 4 models × 4 sparsities × 5 sample sizes × 2 PCA types = 160 experiments
#
# Key questions to answer:
# 1. Does global PCA need more calibration tokens than per-layer?
# 2. Where do returns diminish?
# 3. How does sensitivity vary by model size?

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.config import config
from slicegpt.slicing_scheduler import ConstSlicingScheduler


def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_message(msg: str, log_file: str = None):
    """Log message to console and optionally to file"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"{timestamp} | {msg}"
    print(formatted_msg, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(formatted_msg + "\n")
            f.flush()


def run_single_experiment(
    model_name: str,
    sparsity: float,
    cal_nsamples: int,
    pca_type: str,  # "per-layer" or "global"
    log_file: str = None,
    cal_batch_size: int = 16,
    cal_max_seqlen: int = 2048,
    ppl_eval_batch_size: int = 8,
) -> dict:
    """
    Run a single calibration ablation experiment.
    """
    result = {
        "model": model_name,
        "model_short": model_name.split("/")[-1],
        "sparsity": sparsity,
        "cal_nsamples": cal_nsamples,
        "pca_type": pca_type,
        "status": "running",
        "timestamp_start": datetime.now().isoformat(),
    }
    
    log_message(f"\n{'='*60}", log_file)
    log_message(f"Starting: {result['model_short']}, sparsity={sparsity}, samples={cal_nsamples}, pca={pca_type}", log_file)
    log_message(f"{'='*60}", log_file)
    
    try:
        # Configure
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.dtype = torch.float16
        
        # Load model
        log_message(f"Loading model {model_name}...", log_file)
        model_load_start = time.time()
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
            model_name, None, token=os.getenv('HF_TOKEN', None), dtype=config.dtype
        )
        model = model_adapter.model
        model_load_time = time.time() - model_load_start
        log_message(f"Model loaded in {model_load_time:.1f}s", log_file)
        
        result["num_layers"] = len(model_adapter.get_layers())
        result["hidden_size"] = model_adapter.hidden_size
        result["model_load_time_s"] = model_load_time
        
        # Calculate sliced dimension
        new_embedding_dimension = int((1 - sparsity) * model_adapter.hidden_size)
        new_embedding_dimension -= new_embedding_dimension % 8  # Round to 8
        result["sliced_dim"] = new_embedding_dimension
        
        # Load dataset
        log_message(f"Loading WikiText-2 dataset...", log_file)
        dataset = data_utils.get_dataset("wikitext2")
        train_dataset, test_dataset = dataset["train"], dataset["test"]
        
        # Prepare calibration data with specified number of samples
        log_message(f"Preparing {cal_nsamples} calibration samples (seqlen={cal_max_seqlen})...", log_file)
        train_loader = data_utils.prepare_dataloader(
            dataset=train_dataset,
            tokenizer=tokenizer,
            max_seqlen=cal_max_seqlen,
            batch_size=cal_batch_size,
            nsamples=cal_nsamples,
            varied_seqlen=False,
            seed=42,
        )
        
        # Calculate actual tokens
        total_tokens = cal_nsamples * cal_max_seqlen
        result["cal_tokens"] = total_tokens
        log_message(f"Calibration tokens: {total_tokens:,} ({cal_nsamples} × {cal_max_seqlen})", log_file)
        
        # Prepare test data
        test_loader = data_utils.prepare_test_dataloader(
            dataset=test_dataset, 
            tokenizer=tokenizer, 
            batch_size=ppl_eval_batch_size
        )
        
        # Replace and fuse layers
        log_message("Fusing layers...", log_file)
        layernorm_fusion.replace_layers(model_adapter)
        layernorm_fusion.fuse_modules(model_adapter)
        
        # Record original params
        original_params = sum(int(p.nelement()) for p in model.parameters())
        result["original_params"] = original_params
        
        # Create scheduler
        scheduler = ConstSlicingScheduler(new_embedding_dimension)
        
        # Run rotation and slicing
        log_message(f"Running {pca_type} PCA rotation and slicing...", log_file)
        compress_start = time.time()
        
        if pca_type == "global":
            rotate.rotate_and_slice_global_pca(
                model_adapter, 
                train_loader, 
                scheduler, 
                final_orientation="pca"
            )
        else:  # per-layer
            rotate.rotate_and_slice(
                model_adapter, 
                train_loader, 
                scheduler, 
                final_orientation="pca"
            )
        
        compress_time = time.time() - compress_start
        result["compress_time_s"] = compress_time
        log_message(f"Compression done in {compress_time:.1f}s", log_file)
        
        # Count parameters and shortcuts
        sliced_params = sum(int(p.nelement()) for p in model.parameters())
        result["sliced_params"] = sliced_params
        
        # Count shortcut parameters
        shortcut_params = 0
        attn_shortcuts = 0
        mlp_shortcuts = 0
        for layer_adapter in model_adapter.get_layers():
            layer = layer_adapter.layer
            if hasattr(layer, 'attn_shortcut_Q') and layer.attn_shortcut_Q is not None:
                shortcut_params += layer.attn_shortcut_Q.numel()
                attn_shortcuts += 1
            if hasattr(layer, 'mlp_shortcut_Q') and layer.mlp_shortcut_Q is not None:
                shortcut_params += layer.mlp_shortcut_Q.numel()
                mlp_shortcuts += 1
        
        result["shortcut_params"] = shortcut_params
        result["shortcut_memory_mb"] = shortcut_params * 2 / (1024 * 1024)  # FP16
        result["attn_shortcuts"] = attn_shortcuts
        result["mlp_shortcuts"] = mlp_shortcuts
        
        # Evaluate perplexity
        log_message("Evaluating perplexity...", log_file)
        model.to(config.device)
        eval_start = time.time()
        perplexity = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        eval_time = time.time() - eval_start
        
        result["perplexity"] = float(perplexity)
        result["eval_time_s"] = eval_time
        result["status"] = "success"
        result["total_time_s"] = time.time() - model_load_start
        
        log_message(f"✓ Perplexity: {perplexity:.4f}", log_file)
        log_message(f"  Compress time: {compress_time:.1f}s, Eval time: {eval_time:.1f}s", log_file)
        
        # GPU memory
        if torch.cuda.is_available():
            result["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            log_message(f"  GPU peak memory: {result['gpu_peak_mb']:.0f} MB", log_file)
        
    except torch.cuda.OutOfMemoryError as e:
        result["status"] = "oom"
        result["error"] = str(e)[:500]
        log_message(f"✗ OOM Error: {str(e)[:200]}", log_file)
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
        log_message(f"✗ Error: {e}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)
        
    finally:
        # Cleanup
        try:
            del model, model_adapter, tokenizer, train_loader, test_loader
        except:
            pass
        cleanup_memory()
        result["timestamp_end"] = datetime.now().isoformat()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Full Calibration Token Count Ablation Study")
    parser.add_argument(
        "--models", 
        type=str, 
        nargs="+",
        default=[
            "facebook/opt-125m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "facebook/opt-6.7b",
        ],
        help="Models to test"
    )
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.25, 0.30],
        help="Sparsity levels to test"
    )
    parser.add_argument(
        "--cal-samples",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help="Calibration sample counts to test"
    )
    parser.add_argument(
        "--pca-types",
        type=str,
        nargs="+",
        default=["per-layer", "global"],
        help="PCA types to test"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/calibration_study",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cal-batch-size",
        type=int,
        default=16,
        help="Batch size for calibration"
    )
    parser.add_argument(
        "--cal-max-seqlen",
        type=int,
        default=2048,
        help="Maximum sequence length for calibration"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if exists"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / "logs" / f"calibration_full_study_{timestamp}.log"
    csv_file = output_dir / f"calibration_full_study_{timestamp}.csv"
    checkpoint_file = output_dir / f"checkpoint_{timestamp}.json"
    
    # Check for existing checkpoint to resume
    existing_checkpoint = None
    if args.resume:
        checkpoints = list(output_dir.glob("checkpoint_*.json"))
        if checkpoints:
            latest_cp = max(checkpoints, key=lambda x: x.stat().st_mtime)
            with open(latest_cp) as f:
                existing_checkpoint = json.load(f)
            checkpoint_file = latest_cp
            # Find corresponding CSV
            csv_candidates = list(output_dir.glob("calibration_full_study_*.csv"))
            if csv_candidates:
                csv_file = max(csv_candidates, key=lambda x: x.stat().st_mtime)
            log_message(f"Resuming from checkpoint: {checkpoint_file}", str(log_file))
    
    log_message(f"\n{'='*70}", str(log_file))
    log_message("PHASE 3: FULL CALIBRATION TOKEN COUNT ABLATION STUDY", str(log_file))
    log_message(f"{'='*70}", str(log_file))
    log_message(f"Models: {[m.split('/')[-1] for m in args.models]}", str(log_file))
    log_message(f"Sparsities: {args.sparsities}", str(log_file))
    log_message(f"Calibration samples: {args.cal_samples}", str(log_file))
    log_message(f"PCA types: {args.pca_types}", str(log_file))
    log_message(f"Cal seqlen: {args.cal_max_seqlen}", str(log_file))
    log_message(f"Output: {csv_file}", str(log_file))
    log_message(f"{'='*70}\n", str(log_file))
    
    # Build experiment list - ordered by model (smaller first) for faster initial results
    experiments = []
    for model in args.models:
        for sparsity in args.sparsities:
            for cal_samples in args.cal_samples:
                for pca_type in args.pca_types:
                    exp_id = f"{model.split('/')[-1]}_s{int(sparsity*100)}_n{cal_samples}_{pca_type}"
                    experiments.append({
                        "exp_id": exp_id,
                        "model": model,
                        "sparsity": sparsity,
                        "cal_samples": cal_samples,
                        "pca_type": pca_type,
                    })
    
    total_experiments = len(experiments)
    log_message(f"Total experiments: {total_experiments}", str(log_file))
    
    # Calculate estimated tokens
    for cal_samples in args.cal_samples:
        tokens = cal_samples * args.cal_max_seqlen
        log_message(f"  {cal_samples} samples = {tokens:,} tokens (~{tokens/1000:.0f}K)", str(log_file))
    
    # Load checkpoint if resuming
    completed = set()
    if existing_checkpoint:
        completed = set(existing_checkpoint.get("completed", []))
        log_message(f"Already completed: {len(completed)} experiments", str(log_file))
    
    # CSV columns
    csv_columns = [
        "experiment_id", "model", "model_short", "sparsity", "cal_nsamples", "cal_tokens",
        "pca_type", "status", "perplexity", "compress_time_s", "eval_time_s", "total_time_s",
        "model_load_time_s", "num_layers", "hidden_size", "sliced_dim", "original_params", 
        "sliced_params", "shortcut_params", "shortcut_memory_mb", "attn_shortcuts", "mlp_shortcuts",
        "gpu_peak_mb", "timestamp_start", "timestamp_end", "error"
    ]
    
    # Write CSV header if new file
    if not csv_file.exists():
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
    
    # Run experiments
    results = []
    for i, exp in enumerate(experiments):
        if exp["exp_id"] in completed:
            log_message(f"[{i+1}/{total_experiments}] Skipping {exp['exp_id']} (already done)", str(log_file))
            continue
        
        log_message(f"\n[{i+1}/{total_experiments}] Running {exp['exp_id']}", str(log_file))
        
        result = run_single_experiment(
            model_name=exp["model"],
            sparsity=exp["sparsity"],
            cal_nsamples=exp["cal_samples"],
            pca_type=exp["pca_type"],
            log_file=str(log_file),
            cal_batch_size=args.cal_batch_size,
            cal_max_seqlen=args.cal_max_seqlen,
        )
        
        result["experiment_id"] = exp["exp_id"]
        results.append(result)
        
        # Append to CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            row = {k: result.get(k, "") for k in csv_columns}
            writer.writerow(row)
        
        # Update checkpoint
        completed.add(exp["exp_id"])
        with open(checkpoint_file, "w") as f:
            json.dump({
                "completed": list(completed), 
                "timestamp": datetime.now().isoformat(),
                "total": total_experiments,
                "success": sum(1 for r in results if r.get("status") == "success"),
                "oom": sum(1 for r in results if r.get("status") == "oom"),
                "error": sum(1 for r in results if r.get("status") == "error"),
            }, f, indent=2)
        
        # Force cleanup between experiments
        cleanup_memory()
        time.sleep(2)
    
    # Final Summary
    log_message(f"\n{'='*70}", str(log_file))
    log_message("STUDY COMPLETE", str(log_file))
    log_message(f"{'='*70}", str(log_file))
    
    success = sum(1 for r in results if r.get("status") == "success")
    oom = sum(1 for r in results if r.get("status") == "oom")
    errors = sum(1 for r in results if r.get("status") == "error")
    
    log_message(f"This session: {len(results)} experiments", str(log_file))
    log_message(f"  Success: {success}, OOM: {oom}, Errors: {errors}", str(log_file))
    log_message(f"Total completed: {len(completed)}/{total_experiments}", str(log_file))
    log_message(f"Results saved to: {csv_file}", str(log_file))
    
    # Print summary table grouped by model
    if results:
        log_message("\n=== RESULTS SUMMARY ===", str(log_file))
        current_model = None
        for r in sorted(results, key=lambda x: (x.get('model_short', ''), x.get('cal_nsamples', 0))):
            if r.get("status") == "success":
                model = r.get('model_short')
                if model != current_model:
                    log_message(f"\n--- {model} ---", str(log_file))
                    current_model = model
                log_message(
                    f"  samples={r.get('cal_nsamples'):3d}, {r.get('pca_type'):9s}: "
                    f"PPL={r.get('perplexity', 0):7.2f}, compress={r.get('compress_time_s', 0):6.1f}s",
                    str(log_file)
                )
    
    # Key insights
    log_message("\n=== KEY ANALYSIS QUESTIONS ===", str(log_file))
    log_message("1. Does global PCA need more tokens than per-layer? Compare PPL at same token count", str(log_file))
    log_message("2. Where do returns diminish? Look for PPL plateau as samples increase", str(log_file))
    log_message("3. How does sensitivity vary by model size? Compare PPL deltas across models", str(log_file))


if __name__ == "__main__":
    main()
