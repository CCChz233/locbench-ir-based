#!/usr/bin/env python
"""
评估代码定位结果

用法:
    python evaluate_results.py \
        --loc_outputs new_outputs/ir_function_CodeRankEmbed/loc_outputs.jsonl \
        --dataset_path /workspace/locbench/data/Loc-Bench_V1_dataset.jsonl
"""

import argparse
import sys
from pathlib import Path

# LocAgent 依赖已复制到 IR-based 下，不再需要添加路径
from evaluation.eval_metric import evaluate_results


def main():
    parser = argparse.ArgumentParser(description="评估代码定位结果")
    parser.add_argument("--loc_outputs", type=str, required=True,
                        help="定位输出文件路径（loc_outputs.jsonl）")
    parser.add_argument("--dataset_path", type=str, 
                        default="/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl",
                        help="数据集文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出结果文件路径（可选，默认打印到控制台）")
    
    args = parser.parse_args()
    
    # 定义level2key映射
    level2key_dict = {
        'file': 'found_files',
        'module': 'found_modules',
        'function': 'found_entities'
    }
    
    print("=" * 80)
    print(f"评估文件: {args.loc_outputs}")
    print(f"数据集: {args.dataset_path}")
    print("=" * 80)
    
    # 运行评估
    result = evaluate_results(
        args.loc_outputs,
        level2key_dict,
        dataset_path=args.dataset_path,
        metrics=['acc', 'ndcg', 'precision', 'recall', 'map'],
        k_values_list=[
            [1, 3, 5],  # file level
            [5, 10],   # module level
            [5, 10]    # function level
        ]
    )
    
    # 显示结果
    print("\n评估结果:")
    print(result)
    
    # 保存结果（如果指定了输出文件）
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\n结果已保存到: {args.output}")
    
    print("\n" + "=" * 80)
    
    # 显示关键指标摘要
    print("\n关键指标摘要:")
    print(f"文件级别 - Acc@1: {result['file']['Acc@1'].values[0]:.4f}, Acc@3: {result['file']['Acc@3'].values[0]:.4f}, Acc@5: {result['file']['Acc@5'].values[0]:.4f}")
    print(f"文件级别 - NDCG@1: {result['file']['NDCG@1'].values[0]:.4f}, NDCG@3: {result['file']['NDCG@3'].values[0]:.4f}, NDCG@5: {result['file']['NDCG@5'].values[0]:.4f}")
    print(f"模块级别 - Recall@5: {result['module']['Recall@5'].values[0]:.4f}, Recall@10: {result['module']['Recall@10'].values[0]:.4f}")
    print(f"函数级别 - Recall@5: {result['function']['Recall@5'].values[0]:.4f}, Recall@10: {result['function']['Recall@10'].values[0]:.4f}")


if __name__ == "__main__":
    main()
