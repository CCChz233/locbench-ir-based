# 路径处理：添加IR-based到sys.path（LocAgent依赖已复制到IR-based下）
import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))  # 确保method模块可被导入

import argparse
import logging
import os
import os.path as osp
import pickle
from typing import Iterable, List, Optional

from datasets import load_dataset

from dependency_graph import RepoEntitySearcher
from dependency_graph.build_graph import NODE_TYPE_CLASS, NODE_TYPE_FUNCTION, VERSION
from plugins.location_tools.retriever.bm25_retriever import (
    build_code_retriever_from_repo,
    build_retriever_from_persist_dir,
)
from util.benchmark.setup_repo import setup_repo
from util.utils import append_to_jsonl, load_jsonl
from method.bm25.retriever import BM25Method


def _iter_instances(data, limit: Optional[int]) -> Iterable[dict]:
    if hasattr(data, "select"):
        if limit:
            limit = min(limit, len(data))
            data = data.select(range(limit))
        for instance in data:
            yield instance
    else:
        if limit:
            data = data[:limit]
        for instance in data:
            yield instance


def _problem_text(instance: dict) -> str:
    for key in ("problem_statement", "issue", "description", "prompt", "text"):
        value = instance.get(key)
        if value:
            return value
    return ""


def _load_graph(graph_index_file: str, instance: dict, repo_base_dir: str, build_if_missing: bool):
    if osp.exists(graph_index_file):
        return pickle.load(open(graph_index_file, "rb"))
    if not build_if_missing:
        return None
    repo_dir = setup_repo(instance_data=instance, repo_base_dir=repo_base_dir, dataset=None, split=None)
    from dependency_graph.build_graph import build_graph

    os.makedirs(osp.dirname(graph_index_file), exist_ok=True)
    graph = build_graph(repo_dir, global_import=True)
    with open(graph_index_file, "wb") as f:
        pickle.dump(graph, f)
    return graph


def _load_retriever(persist_dir: str, instance: dict, repo_base_dir: str, build_if_missing: bool):
    if osp.exists(osp.join(persist_dir, "corpus.jsonl")):
        return build_retriever_from_persist_dir(persist_dir)
    if not build_if_missing:
        return None
    repo_dir = setup_repo(instance_data=instance, repo_base_dir=repo_base_dir, dataset=None, split=None)
    os.makedirs(persist_dir, exist_ok=True)
    return build_code_retriever_from_repo(repo_dir, persist_path=persist_dir)


def _module_id(entity_id: str) -> str:
    file_path, name = entity_id.split(":", 1)
    if "." in name:
        name = name.split(".")[0]
    return f"{file_path}:{name}"


def _instance_id_to_repo_name(instance_id: str) -> str:
    """Convert instance_id (e.g., 'UXARRAY__uxarray-1117') to repo_name (e.g., 'UXARRAY_uxarray')."""
    # Remove issue number suffix (e.g., '-1117')
    import re
    repo_part = re.sub(r'-\d+$', '', instance_id)
    # Replace double underscore with single underscore
    return repo_part.replace('__', '_')


def _dedupe_append(target: List[str], item: str, limit: int) -> None:
    if item not in target and len(target) < limit:
        target.append(item)


def _clean_file_path(file_path: str, repo_name: str) -> str:
    """Clean file path to remove absolute path prefix and keep relative path."""
    # Handle paths like 'workspace/LocAgent//uxarray/grid/grid.py'
    # or '/workspace/LocAgent/playground/locbench_repos/UXARRAY_uxarray/uxarray/...'
    import re
    # Try to extract the relative path after the repo directory
    # Pattern: anything ending with org_repo/ followed by the actual relative path
    repo_pattern = repo_name.replace('_', '[_/]')  # Match both underscore and slash
    match = re.search(rf'{repo_pattern}/(.+)$', file_path, re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback: just get the path after double slash or after last known prefix
    if '//' in file_path:
        return file_path.split('//')[-1]
    return file_path


def run_instance(
    instance: dict,
    graph_index_dir: str,
    bm25_index_dir: str,
    repo_base_dir: str,
    build_if_missing: bool,
    top_k_files: int,
    top_k_modules: int,
    top_k_entities: int,
    mapper_type: str = "graph",
    repos_root: Optional[str] = None,
):
    """
    运行BM25定位实例（使用BM25Method类）
    
    Args:
        instance: 数据集实例
        graph_index_dir: Graph索引目录
        bm25_index_dir: BM25索引目录
        repo_base_dir: 仓库基础目录（用于构建缺失的索引）
        build_if_missing: 如果索引缺失是否构建
        top_k_files: 返回的文件数量
        top_k_modules: 返回的模块数量
        top_k_entities: 返回的实体数量
        mapper_type: 映射器类型，'graph' 或 'ast'
        repos_root: 源代码仓库根目录（mapper_type='ast'时必需）
    
    Returns:
        结果字典或None
    """
    try:
        # 使用BM25Method类进行定位
        method = BM25Method(
            graph_index_dir=graph_index_dir,
            bm25_index_dir=bm25_index_dir,
            top_k_files=top_k_files,
            top_k_modules=top_k_modules,
            top_k_entities=top_k_entities,
            mapper_type=mapper_type,
            repos_root=repos_root,
        )
        
        result = method.localize(instance)
        
        # 转换为字典格式（兼容原有输出格式）
        return {
            "instance_id": result.instance_id,
            "found_files": result.found_files,
            "found_modules": result.found_modules,
            "found_entities": result.found_entities,
            "raw_output_loc": [],
        }
    except Exception as e:
        logging.warning("Error processing instance %s: %s", instance.get("instance_id"), e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Run BM25 baseline localization.")
    parser.add_argument("--dataset", type=str, default="czlll/Loc-Bench_V1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument("--graph_index_dir", type=str, default="")
    parser.add_argument("--bm25_index_dir", type=str, default="")
    parser.add_argument("--repo_base_dir", type=str, default="playground/bm25_baseline")
    parser.add_argument("--eval_n_limit", type=int, default=0)
    parser.add_argument("--top_k_files", type=int, default=15)
    parser.add_argument("--top_k_modules", type=int, default=15)
    parser.add_argument("--top_k_entities", type=int, default=15)
    parser.add_argument("--build_if_missing", action="store_true")
    parser.add_argument(
        "--mapper_type",
        type=str,
        choices=["graph", "ast"],
        default="graph",
        help="映射器类型: 'graph' (Graph索引+span_ids, 默认) 或 'ast' (AST解析)"
    )
    parser.add_argument(
        "--repos_root",
        type=str,
        default="",
        help="源代码仓库根目录（使用 --mapper_type ast 时必需）"
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name or args.dataset.split("/")[-1]
    if args.dataset_path and not args.dataset_name:
        dataset_name = "Loc-Bench_V1"

    graph_index_dir = (
        args.graph_index_dir
        or os.environ.get("GRAPH_INDEX_DIR")
        or f"index_data/{dataset_name}/graph_index_{VERSION}"
    )
    bm25_index_dir = (
        args.bm25_index_dir
        or os.environ.get("BM25_INDEX_DIR")
        or f"index_data/{dataset_name}/BM25_index"
    )

    os.makedirs(args.output_folder, exist_ok=True)
    output_file = osp.join(args.output_folder, args.output_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.dataset_path:
        data = load_jsonl(args.dataset_path)
    else:
        data = load_dataset(args.dataset, split=args.split)

    processed = set()
    if osp.exists(output_file):
        for row in load_jsonl(output_file):
            processed.add(row.get("instance_id"))

    for instance in _iter_instances(data, args.eval_n_limit or None):
        instance_id = instance.get("instance_id")
        if not instance_id or instance_id in processed:
            continue
        result = run_instance(
            instance=instance,
            graph_index_dir=graph_index_dir,
            bm25_index_dir=bm25_index_dir,
            repo_base_dir=args.repo_base_dir,
            build_if_missing=args.build_if_missing,
            top_k_files=args.top_k_files,
            top_k_modules=args.top_k_modules,
            top_k_entities=args.top_k_entities,
            mapper_type=args.mapper_type,
            repos_root=args.repos_root if args.repos_root else None,
        )
        if result:
            append_to_jsonl(result, output_file)


if __name__ == "__main__":
    main()
