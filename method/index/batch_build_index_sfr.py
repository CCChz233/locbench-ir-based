#!/usr/bin/env python
"""
SFR 专用索引构建工具
使用 SentenceTransformer 后端
"""

import argparse
import json
import os
import os.path as osp
import random
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
from multiprocessing import Value

# 导入共享代码
from batch_build_index_common import (
    Block,
    BlockDataset,
    # 所有切块策略
    collect_ir_function_blocks,
    collect_function_blocks,
    collect_llamaindex_code_blocks,
    collect_llamaindex_sentence_blocks,
    collect_llamaindex_token_blocks,
    collect_llamaindex_semantic_blocks,
    collect_langchain_fixed_blocks,
    collect_langchain_recursive_blocks,
    collect_langchain_token_blocks,
    collect_epic_blocks,
    # 工具函数
    resolve_model_name,
    list_folders,
    instance_id_to_repo_name,
    save_index,
    EpicSplitterConfig,
)

# 尝试导入 datasets，如果失败则只支持本地模式
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def embed_blocks(
    blocks: List[Block],
    model: SentenceTransformer,  # 明确类型
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """SFR 专用的 embedding 函数，使用 SentenceTransformer 后端"""
    ds = BlockDataset(blocks, tokenizer, max_length, ir_context_tokens=256)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    outs = []

    with torch.no_grad():
        for input_ids, attn_mask, orig_len, truncated, is_ir in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            # SFR 使用 SentenceTransformer 的调用方式
            outputs = model.forward({
                "input_ids": input_ids,
                "attention_mask": attn_mask
            })

            # 处理 SentenceTransformer 输出
            if isinstance(outputs, dict):
                sent_emb = outputs.get("sentence_embedding") or outputs.get("embeddings")
            elif isinstance(outputs, torch.Tensor):
                sent_emb = outputs
            else:
                raise RuntimeError(f"Unexpected output type: {type(outputs)}")

            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            outs.append(sent_emb.cpu())

    return torch.cat(outs, dim=0)


def monitor_progress(completed_repos: Value, total_pbar: tqdm, total_repos: int):
    """监控总进度并更新进度条"""
    import time
    while True:
        current = completed_repos.value
        total_pbar.n = current
        total_pbar.refresh()
        if current >= total_repos:
            break
        time.sleep(1)


def run(rank: int, repo_queue, args, gpu_ids: list, total_repos: int, completed_repos: Value):
    """单个进程的工作函数"""
    import signal
    import atexit

    if gpu_ids and not args.force_cpu:
        actual_gpu_id = gpu_ids[rank % len(gpu_ids)]
        device = torch.device(f'cuda:{actual_gpu_id}')
        print(f'[Process {rank}] Using GPU {actual_gpu_id}')
    else:
        device = torch.device('cpu')
        print(f'[Process {rank}] Using CPU')

    def cleanup_resources():
        """清理资源"""
        try:
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            torch.cuda.empty_cache()
        except:
            pass

    atexit.register(cleanup_resources)

    while True:
        try:
            repo_info = repo_queue.get(timeout=1)
        except:
            break

        if repo_info is None:
            break

        repo_path, repo_name = repo_info
        print(f'[Process {rank}] Processing repo: {repo_name}')

        try:
            # 根据策略收集代码块
            if args.strategy == "ir_function":
                blocks, function_metadata = collect_ir_function_blocks(repo_path)
            elif args.strategy == "function_level":
                blocks, function_metadata = collect_function_blocks(repo_path, args.block_size)
            elif args.strategy == "llamaindex_code":
                blocks = collect_llamaindex_code_blocks(repo_path)
                function_metadata = {}
            elif args.strategy == "llamaindex_sentence":
                blocks = collect_llamaindex_sentence_blocks(repo_path)
                function_metadata = {}
            elif args.strategy == "llamaindex_token":
                blocks = collect_llamaindex_token_blocks(repo_path)
                function_metadata = {}
            elif args.strategy == "llamaindex_semantic":
                blocks = collect_llamaindex_semantic_blocks(repo_path)
                function_metadata = {}
            elif args.strategy == "langchain_fixed":
                blocks = collect_langchain_fixed_blocks(repo_path)
                function_metadata = {}
            elif args.strategy == "langchain_recursive":
                blocks = collect_langchain_recursive_blocks(repo_path)
                function_metadata = {}
            elif args.strategy == "langchain_token":
                blocks = collect_langchain_token_blocks(repo_path)
                function_metadata = {}
            elif args.strategy == "epic":
                blocks = collect_epic_blocks(repo_path)
                function_metadata = {}
            else:
                raise ValueError(f"Unknown strategy: {args.strategy}")

            if not blocks:
                print(f'[Process {rank}] No blocks collected from {repo_name}, skipping')
                continue

            print(f'[Process {rank}] Collected {len(blocks)} blocks from {repo_name}')

            # 加载 SFR 模型（固定使用 SentenceTransformer）
            print(f'[Process {rank}] Loading SFR model on GPU {actual_gpu_id}...')
            trust_remote_code = args.trust_remote_code
            model_name = resolve_model_name(args.model_name)

            try:
                model = SentenceTransformer(
                    model_name,
                    device=device,
                    trust_remote_code=trust_remote_code
                )
                tokenizer = model.tokenizer
                print(f'[Process {rank}] Model loaded on GPU {actual_gpu_id}.')
            except Exception as e:
                print(f'[Process {rank}] Failed to load model: {e}')
                cleanup_resources()
                return

            # 生成嵌入
            embeddings = embed_blocks(
                blocks, model, tokenizer,
                args.max_length, args.batch_size, device,
            )

            # 保存索引
            index_dir = Path(args.index_dir) / f"dense_index_{args.strategy}" / repo_name
            index_dir.mkdir(parents=True, exist_ok=True)

            save_index(
                index_dir,
                embeddings,
                blocks,
                args.strategy,
                function_metadata=function_metadata,
            )

            print(f'[Process {rank}] Saved index for {repo_name} with {len(blocks)} blocks')

            # 更新完成计数器
            with completed_repos.get_lock():
                completed_repos.value += 1

        except Exception as e:
            print(f'[Process {rank}] Error processing {repo_name}: {e}')
            import traceback
            traceback.print_exc()

            # 即使出错也更新计数器，避免进度条卡住
            with completed_repos.get_lock():
                completed_repos.value += 1

        finally:
            cleanup_resources()


def main():
    parser = argparse.ArgumentParser(
        description="SFR 专用索引构建工具（使用 SentenceTransformer）"
    )

    # 数据源参数
    parser.add_argument("--dataset", type=str, default="",
                       help="HuggingFace dataset name (e.g., 'czlll/Loc-Bench_V1')")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split")
    parser.add_argument("--repo_path", type=str,
                       default="/workspace/locbench/repos/locbench_repos",
                       help="Local repository directory")
    parser.add_argument("--index_dir", type=str,
                       default="/workspace/locbench/IR-based/new_index_data",
                       help="Output directory for indexes")
    parser.add_argument("--instance_id_path", type=str, default="",
                       help="Path to instance_id file for filtering")

    # 模型参数（SFR 专用）
    parser.add_argument("--model_name", type=str, required=True,
                       help="SFR model path")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Allow execution of model repository code")

    # 切块策略参数
    parser.add_argument("--strategy", type=str, required=True,
                       choices=[
                           "ir_function", "function_level",
                           "llamaindex_code", "llamaindex_sentence", "llamaindex_token", "llamaindex_semantic",
                           "langchain_fixed", "langchain_recursive", "langchain_token",
                           "epic"
                       ],
                       help="Chunking strategy")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for embedding")
    parser.add_argument("--block_size", type=int, default=15,
                       help="Block size for fixed chunking")
    parser.add_argument("--ir_function_context_tokens", type=int, default=256,
                       help="Context tokens for IR function strategy")

    # 并行处理参数
    parser.add_argument("--num_processes", type=int, default=4,
                       help="Number of parallel processes")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3",
                       help="Comma-separated GPU IDs")
    parser.add_argument("--force_cpu", action="store_true",
                       help="Force CPU usage")

    args = parser.parse_args()

    # 设置 GPU
    gpu_ids = []  # 初始化为空列表
    if not args.force_cpu:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip()]
        if not gpu_ids:
            print("Warning: No valid GPU IDs provided, falling back to CPU")
            args.force_cpu = True
        else:
            # 检测实际可用的 GPU 数量
            available_gpus = torch.cuda.device_count()
            if available_gpus == 0:
                print("Warning: No CUDA devices available, falling back to CPU")
                args.force_cpu = True
                gpu_ids = []
            else:
                # 限制使用的 GPU 数量不超过实际可用数
                max_gpu = max(gpu_ids) + 1
                if max_gpu > available_gpus:
                    print(f"Warning: Requested GPUs {gpu_ids} but only {available_gpus} available")
                    print(f"Limiting to use GPUs 0-{available_gpus-1}")
                    gpu_ids = [i for i in range(available_gpus)]

    # 获取仓库列表
    if args.dataset and HAS_DATASETS:
        print(f"Loading dataset {args.dataset}")
        ds = load_dataset(args.dataset, split=args.split)
        repos = []
        for item in ds:
            repo_name = item.get('repo_name') or item.get('instance_id', '').split('-')[0]
            if repo_name:
                repos.append(repo_name)
    else:
        print(f"Using local repos from {args.repo_path}")
        repos = list_folders(args.repo_path)

    if args.instance_id_path:
        try:
            with open(args.instance_id_path, 'r') as f:
                instance_ids = [line.strip() for line in f if line.strip()]
            repos = [instance_id_to_repo_name(iid) for iid in instance_ids if iid]
        except Exception as e:
            print(f"Warning: Failed to load instance_ids from {args.instance_id_path}: {e}")

    repos = list(set(repos))  # 去重
    print(f"Processing {len(repos)} repositories")

    if not repos:
        print("No repositories to process")
        return

    # 创建多进程共享计数器和总进度条
    completed_repos = Value('i', 0)
    total_pbar = tqdm(
        total=len(repos),
        desc="总进度",
        ncols=80,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

    # 启动进度监控线程
    monitor_thread = threading.Thread(
        target=monitor_progress,
        args=(completed_repos, total_pbar, len(repos)),
        daemon=True
    )
    monitor_thread.start()

    # 创建仓库队列
    repo_queue = mp.Queue()
    for repo_name in repos:
        if args.dataset and HAS_DATASETS:
            repo_path = osp.join(args.repo_path, repo_name)
        else:
            repo_path = osp.join(args.repo_path, repo_name)
        repo_queue.put((repo_path, repo_name))

    # 启动多进程
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=run, args=(rank, repo_queue, args, gpu_ids, len(repos), completed_repos))
        p.start()
        processes.append(p)

    # 等待完成
    for p in processes:
        p.join()

    # 确保进度条完成
    total_pbar.n = len(repos)
    total_pbar.refresh()
    total_pbar.close()

    print("All processes completed")


if __name__ == "__main__":
    main()