#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# LLaMA-2-7B Full Study
# Runs all experiments: Global PCA, Throughput, K-Block, Calibration

import argparse
import csv
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.config import config
from slicegpt.slicing_scheduler import ConstSlicingScheduler


MODEL_NAME = "meta-llama/Llama-2-7b-hf"
MODEL_SHORT = "llama-2-7b"


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def log_message(msg: str, log_file: str = None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"{timestamp} | {msg}"
    print(formatted_msg, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(formatted_msg + "\n")
            f.flush()


def measure_throughput(model, tokenizer, seq_len=128, batch_size=1, num_runs=30, warmup=5):
    """Measure prefill and decode throughput"""
    import statistics
    device = next(model.parameters()).device
    
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids, attention_mask=attention_mask)
    
    torch.cuda.synchronize()
    
    # Prefill
    prefill_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
            torch.cuda.synchronize()
            prefill_times.append(time.perf_counter() - start)
    
    # Decode
    decode_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
            past = outputs.past_key_values
            next_token = torch.randint(0, tokenizer.vocab_size, (batch_size, 1), device=device)
            new_mask = torch.ones((batch_size, seq_len + 1), device=device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(next_token, attention_mask=new_mask, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - start)
    
    prefill_median = statistics.median(prefill_times)
    decode_median = statistics.median(decode_times)
    
    return {
        "prefill_median_ms": prefill_median * 1000,
        "prefill_tokens_per_sec": (batch_size * seq_len) / prefill_median,
        "decode_median_ms": decode_median * 1000,
        "decode_tokens_per_sec": batch_size / decode_median,
    }


def run_pca_experiment(sparsity: float, pca_type: str, log_file: str, 
                       cal_nsamples: int = 128, measure_throughput_flag: bool = False):
    """Run Global PCA or Per-layer experiment"""
    
    result = {
        "model": MODEL_NAME,
        "model_short": MODEL_SHORT,
        "sparsity": sparsity,
        "pca_type": pca_type,
        "cal_nsamples": cal_nsamples,
        "status": "running",
        "timestamp_start": datetime.now().isoformat(),
    }
    
    log_message(f"\n{'='*60}", log_file)
    log_message(f"{MODEL_SHORT}: sparsity={sparsity}, pca={pca_type}, samples={cal_nsamples}", log_file)
    log_message(f"{'='*60}", log_file)
    
    try:
        config.device = torch.device("cuda")
        config.dtype = torch.float16
        
        # Load model
        log_message("Loading LLaMA-2-7B...", log_file)
        hf_token = os.getenv('HF_TOKEN', None)
        if not hf_token:
            token_file = Path("/home/danish/opus_slicegpt/hf_token.txt")
            if token_file.exists():
                hf_token = token_file.read_text().strip()
        
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
            MODEL_NAME, None, token=hf_token, dtype=config.dtype
        )
        model = model_adapter.model
        log_message("Model loaded", log_file)
        
        result["num_layers"] = len(model_adapter.get_layers())
        result["hidden_size"] = model_adapter.hidden_size
        
        # Sliced dimension
        new_dim = int((1 - sparsity) * model_adapter.hidden_size)
        new_dim -= new_dim % 8
        result["sliced_dim"] = new_dim
        
        # Original params
        original_params = sum(int(p.nelement()) for p in model.parameters())
        result["original_params"] = original_params
        
        # Load data
        dataset = data_utils.get_dataset("wikitext2")
        train_loader = data_utils.prepare_dataloader(
            dataset=dataset["train"],
            tokenizer=tokenizer,
            max_seqlen=2048,
            batch_size=8,  # Smaller for LLaMA
            nsamples=cal_nsamples,
            varied_seqlen=False,
            seed=42,
        )
        test_loader = data_utils.prepare_test_dataloader(
            dataset=dataset["test"],
            tokenizer=tokenizer,
            batch_size=4
        )
        
        # Fuse
        log_message("Fusing layers...", log_file)
        layernorm_fusion.replace_layers(model_adapter)
        layernorm_fusion.fuse_modules(model_adapter)
        
        # Compress
        scheduler = ConstSlicingScheduler(new_dim)
        
        log_message(f"Running {pca_type} compression...", log_file)
        compress_start = time.time()
        
        if pca_type == "global":
            rotate.rotate_and_slice_global_pca(model_adapter, train_loader, scheduler, final_orientation="pca")
        else:
            rotate.rotate_and_slice(model_adapter, train_loader, scheduler, final_orientation="pca")
        
        compress_time = time.time() - compress_start
        result["compress_time_s"] = compress_time
        log_message(f"Compression done in {compress_time:.1f}s", log_file)
        
        # Count params
        sliced_params = sum(int(p.nelement()) for p in model.parameters())
        result["sliced_params"] = sliced_params
        
        # Shortcuts
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
        result["shortcut_memory_mb"] = shortcut_params * 2 / (1024 * 1024)
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
        log_message(f"✓ Perplexity: {perplexity:.4f}", log_file)
        
        # Throughput if requested
        if measure_throughput_flag:
            log_message("Measuring throughput...", log_file)
            model.eval()
            throughput = measure_throughput(model, tokenizer)
            result.update(throughput)
            log_message(f"✓ Prefill: {throughput['prefill_tokens_per_sec']:.0f} tok/s", log_file)
            log_message(f"✓ Decode: {throughput['decode_tokens_per_sec']:.1f} tok/s", log_file)
        
        result["status"] = "success"
        
        if torch.cuda.is_available():
            result["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
    except torch.cuda.OutOfMemoryError as e:
        result["status"] = "oom"
        result["error"] = str(e)[:500]
        log_message(f"✗ OOM Error", log_file)
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
        log_message(f"✗ Error: {e}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)
        
    finally:
        try:
            del model, model_adapter, tokenizer
        except:
            pass
        cleanup_memory()
        result["timestamp_end"] = datetime.now().isoformat()
    
    return result


def run_global_pca_study(output_dir: Path, log_file: str):
    """Run Global PCA study - compare per-layer vs global"""
    log_message("\n" + "="*70, log_file)
    log_message("STUDY 1: GLOBAL PCA vs PER-LAYER", log_file)
    log_message("="*70, log_file)
    
    csv_file = output_dir / "global_pca_study_llama.csv"
    
    columns = [
        "model", "model_short", "sparsity", "pca_type", "status",
        "perplexity", "compress_time_s", "eval_time_s",
        "original_params", "sliced_params", "shortcut_params", "shortcut_memory_mb",
        "attn_shortcuts", "mlp_shortcuts", "num_layers", "hidden_size", "sliced_dim",
        "gpu_peak_mb", "timestamp_start", "timestamp_end", "error"
    ]
    
    with open(csv_file, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=columns).writeheader()
    
    sparsities = [0.10, 0.20, 0.25, 0.30]
    pca_types = ["per-layer", "global"]
    
    for sparsity in sparsities:
        for pca_type in pca_types:
            result = run_pca_experiment(sparsity, pca_type, log_file)
            
            with open(csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writerow({k: result.get(k, "") for k in columns})
            
            cleanup_memory()
            time.sleep(3)
    
    log_message(f"✓ Global PCA study saved to: {csv_file}", log_file)


def run_throughput_study(output_dir: Path, log_file: str):
    """Run Throughput study"""
    log_message("\n" + "="*70, log_file)
    log_message("STUDY 2: THROUGHPUT (tokens/sec)", log_file)
    log_message("="*70, log_file)
    
    csv_file = output_dir / "throughput_study_llama.csv"
    
    columns = [
        "model", "model_short", "sparsity", "pca_type", "status",
        "prefill_median_ms", "prefill_tokens_per_sec",
        "decode_median_ms", "decode_tokens_per_sec",
        "compress_time_s", "shortcut_params", "shortcut_memory_mb",
        "original_params", "sliced_params", "gpu_peak_mb",
        "timestamp_start", "timestamp_end", "error"
    ]
    
    with open(csv_file, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=columns).writeheader()
    
    sparsities = [0.10, 0.20, 0.25, 0.30]
    pca_types = ["per-layer", "global"]
    
    for sparsity in sparsities:
        for pca_type in pca_types:
            result = run_pca_experiment(sparsity, pca_type, log_file, measure_throughput_flag=True)
            
            with open(csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writerow({k: result.get(k, "") for k in columns})
            
            cleanup_memory()
            time.sleep(3)
    
    log_message(f"✓ Throughput study saved to: {csv_file}", log_file)


def run_kblock_study(output_dir: Path, log_file: str):
    """Run K-Block study - only K=1 (per-layer) and K=L (global) based on OPT findings"""
    log_message("\n" + "="*70, log_file)
    log_message("STUDY 3: K-BLOCK (K=1 vs K=L only, based on OPT findings)", log_file)
    log_message("="*70, log_file)
    
    # Based on OPT findings, intermediate K values don't work
    # Just verify this holds for LLaMA too
    log_message("Note: OPT study showed intermediate K values fail catastrophically", log_file)
    log_message("Testing only K=1 (per-layer) and K=L (global) for LLaMA", log_file)
    
    csv_file = output_dir / "kblock_study_llama.csv"
    
    columns = [
        "model", "model_short", "sparsity", "k_value", "method", "status",
        "perplexity", "compress_time_s", "shortcut_params", "shortcut_memory_mb",
        "timestamp_start", "timestamp_end", "error"
    ]
    
    with open(csv_file, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=columns).writeheader()
    
    sparsities = [0.10, 0.20, 0.25, 0.30]
    
    for sparsity in sparsities:
        # K=1 (per-layer)
        result = run_pca_experiment(sparsity, "per-layer", log_file)
        result["k_value"] = 1
        result["method"] = "per-layer"
        
        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writerow({k: result.get(k, "") for k in columns})
        
        cleanup_memory()
        time.sleep(3)
        
        # K=L (global)
        result = run_pca_experiment(sparsity, "global", log_file)
        result["k_value"] = 32  # LLaMA-2-7B has 32 layers
        result["method"] = "global"
        
        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writerow({k: result.get(k, "") for k in columns})
        
        cleanup_memory()
        time.sleep(3)
    
    log_message(f"✓ K-Block study saved to: {csv_file}", log_file)


def run_calibration_study(output_dir: Path, log_file: str):
    """Run Calibration token count study"""
    log_message("\n" + "="*70, log_file)
    log_message("STUDY 4: CALIBRATION TOKEN COUNT", log_file)
    log_message("="*70, log_file)
    
    csv_file = output_dir / "calibration_study_llama.csv"
    
    columns = [
        "model", "model_short", "sparsity", "cal_nsamples", "pca_type", "status",
        "perplexity", "compress_time_s", "eval_time_s",
        "shortcut_params", "shortcut_memory_mb",
        "timestamp_start", "timestamp_end", "error"
    ]
    
    with open(csv_file, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=columns).writeheader()
    
    sparsities = [0.25]  # Focus on 25% sparsity for calibration study
    cal_samples_list = [16, 32, 64, 128, 256]
    pca_types = ["per-layer", "global"]
    
    for sparsity in sparsities:
        for cal_samples in cal_samples_list:
            for pca_type in pca_types:
                result = run_pca_experiment(sparsity, pca_type, log_file, cal_nsamples=cal_samples)
                
                with open(csv_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writerow({k: result.get(k, "") for k in columns})
                
                cleanup_memory()
                time.sleep(3)
    
    log_message(f"✓ Calibration study saved to: {csv_file}", log_file)


def main():
    parser = argparse.ArgumentParser(description="LLaMA-2-7B Full Study")
    parser.add_argument("--study", choices=["all", "pca", "throughput", "kblock", "calibration"], 
                        default="all", help="Which study to run")
    parser.add_argument("--output-dir", default="results/llama_study")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = str(output_dir / f"llama_study_{timestamp}.log")
    
    log_message("="*70, log_file)
    log_message("LLaMA-2-7B FULL STUDY", log_file)
    log_message("="*70, log_file)
    log_message(f"Model: {MODEL_NAME}", log_file)
    log_message(f"Study: {args.study}", log_file)
    log_message(f"Output: {output_dir}", log_file)
    
    if args.study in ["all", "pca"]:
        run_global_pca_study(output_dir, log_file)
    
    if args.study in ["all", "throughput"]:
        run_throughput_study(output_dir, log_file)
    
    if args.study in ["all", "kblock"]:
        run_kblock_study(output_dir, log_file)
    
    if args.study in ["all", "calibration"]:
        run_calibration_study(output_dir, log_file)
    
    log_message("\n" + "="*70, log_file)
    log_message("ALL STUDIES COMPLETE", log_file)
    log_message("="*70, log_file)


if __name__ == "__main__":
    main()
