# 路径处理：添加IR-based到sys.path（LocAgent依赖已复制到IR-based下）
import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))  # 确保method模块可被导入

import argparse
import json
import os
import subprocess
import time
from typing import Any, Dict, List

import pandas as pd
from evaluation.eval_metric import evaluate_results


def _flatten_eval(df: pd.DataFrame) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    row = df.iloc[0]
    for level, metric in df.columns:
        flat[f"{level}_{metric}"] = row[(level, metric)]
    return flat


def run_bm25(method: Dict[str, Any], dataset_path: str, eval_n_limit: int | None) -> None:
    cmd = [
        "python",
        "scripts/run_bm25_baseline.py",
        "--dataset_path",
        dataset_path,
        "--output_folder",
        method["output_folder"],
        "--graph_index_dir",
        method["graph_index_dir"],
        "--bm25_index_dir",
        method["bm25_index_dir"],
    ]
    if eval_n_limit:
        cmd += ["--eval_n_limit", str(eval_n_limit)]
    subprocess.run(cmd, check=True)


def run_locator(method: Dict[str, Any], dataset_path: str, repos_root: str, eval_n_limit: int | None) -> None:
    cmd = [
        "python",
        "method/RepoCoder/run_locator.py",
        "--dataset_path",
        dataset_path,
        "--output_folder",
        method["output_folder"],
        "--mode",
        method.get("mode", "dense"),
        "--block_size",
        str(method.get("block_size", 15)),
        "--top_k_blocks",
        str(method.get("top_k_blocks", 50)),
        "--top_k_files",
        str(method.get("top_k_files", 10)),
    ]
    if repos_root:
        cmd += ["--repos_root", repos_root]
    elif method.get("repo_path"):
        cmd += ["--repo_path", method["repo_path"]]
    else:
        raise ValueError("Either repos_root or method.repo_path must be provided for locator.")

    if method.get("model_name"):
        cmd += ["--model_name", method["model_name"]]
    if method.get("max_length"):
        cmd += ["--max_length", str(method["max_length"])]
    if method.get("batch_size"):
        cmd += ["--batch_size", str(method["batch_size"])]
    if method.get("max_blocks_per_file") is not None:
        cmd += ["--max_blocks_per_file", str(method["max_blocks_per_file"])]
    if eval_n_limit:
        cmd += ["--eval_n_limit", str(eval_n_limit)]
    subprocess.run(cmd, check=True)


def evaluate_and_save(output_folder: str, dataset_path: str) -> Dict[str, float]:
    output_file = os.path.join(output_folder, "loc_outputs.jsonl")
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Missing output file: {output_file}")
    level2key = {"file": "found_files", "module": "found_modules", "function": "found_entities"}
    df = evaluate_results(output_file, level2key, dataset_path=dataset_path)
    eval_path = os.path.join(output_folder, "eval_results.csv")
    df.to_csv(eval_path, index=False)
    return _flatten_eval(df)


def main():
    parser = argparse.ArgumentParser(description="Run retrieval benchmarks across multiple methods.")
    parser.add_argument("--config", default="configs/retrieval_benchmark.json", help="Path to config JSON.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dataset_path = cfg["dataset_path"]
    repos_root = cfg.get("repos_root", "")
    eval_n_limit = cfg.get("eval_n_limit", 0) or None
    methods = cfg["methods"]

    compare_rows: List[Dict[str, Any]] = []
    for method in methods:
        name = method["name"]
        method_type = method["type"]
        output_folder = method["output_folder"]
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        print(f"[{name}] Running method type={method_type}")
        if method_type == "bm25":
            run_bm25(method, dataset_path, eval_n_limit)
        elif method_type == "locator":
            run_locator(method, dataset_path, repos_root, eval_n_limit)
        else:
            raise ValueError(f"Unsupported method type: {method_type}")

        print(f"[{name}] Evaluating results...")
        metrics = evaluate_and_save(output_folder, dataset_path)
        compare_row = {"method": name}
        compare_row.update(metrics)
        compare_rows.append(compare_row)
        print(f"[{name}] Done.")

    if compare_rows:
        compare_df = pd.DataFrame(compare_rows)
        ts = time.strftime("%Y%m%d-%H%M%S")
        compare_path = cfg.get("compare_output", f"outputs/compare_{ts}.csv")
        Path(compare_path).parent.mkdir(parents=True, exist_ok=True)
        compare_df.to_csv(compare_path, index=False)
        print(f"Saved comparison to {compare_path}")


if __name__ == "__main__":
    main()
