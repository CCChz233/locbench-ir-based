# IR-based 快速开始指南

## 1. 环境准备

```bash
# 激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh  # 如果需要
conda activate locagent

# 进入 IR-based 目录
cd /workspace/locbench/IR-based

# 设置环境变量（可选，脚本已自动处理路径）
export PYTHONPATH="/workspace/locbench/IR-based:/workspace/locbench/LocAgent:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
```

## 2. 运行 BM25 基线（推荐，最快上手）

### 测试模式（运行前10个实例）

```bash
cd /workspace/locbench/IR-based

python scripts/run_bm25_baseline.py \
    --dataset_path ../data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/bm25_test \
    --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \
    --bm25_index_dir index_data/Loc-Bench_V1/BM25_index \
    --top_k_files 10 \
    --top_k_modules 10 \
    --top_k_entities 50 \
    --eval_n_limit 10
```

### 完整运行（全部560个实例）

```bash
cd /workspace/locbench/IR-based

python scripts/run_bm25_baseline.py \
    --dataset_path ../data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/bm25_results \
    --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \
    --bm25_index_dir index_data/Loc-Bench_V1/BM25_index \
    --top_k_files 10 \
    --top_k_modules 10 \
    --top_k_entities 50
```

## 3. 评估结果

运行完成后，评估结果：

```bash
cd /workspace/locbench/IR-based

python -c "
import sys
sys.path.insert(0, '../LocAgent')
from evaluation.eval_metric import evaluate_results

level2key = {
    'file': 'found_files',
    'module': 'found_modules',
    'function': 'found_entities'
}

results = evaluate_results(
    'outputs/bm25_test/loc_outputs.jsonl',
    level2key,
    dataset_path='../data/Loc-Bench_V1_dataset.jsonl'
)
print(results)
"
```

## 4. 其他方法

### Dense 检索

```bash
cd /workspace/locbench/IR-based

python method/dense/run_with_index.py \
    --index_dir index_data/Loc-Bench_V1/dense_index_fixed \
    --dataset_path ../data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/dense_results \
    --model_name models/rlretriever \
    --repos_root ../repos/locbench_repos \
    --top_k_files 10 \
    --top_k_modules 10 \
    --top_k_entities 50 \
    --eval_n_limit 10  # 测试用，删除此参数运行全部
```

### Jaccard 检索

```bash
cd /workspace/locbench/IR-based

python method/RepoCoder/run_locator.py \
    --repos_root ../repos/locbench_repos \
    --dataset_path ../data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/jaccard_results \
    --mode jaccard \
    --block_size 15 \
    --top_k_blocks 50 \
    --top_k_files 10
```

## 5. 查看帮助

```bash
cd /workspace/locbench/IR-based

# 查看 BM25 脚本帮助
python scripts/run_bm25_baseline.py --help

# 查看索引构建脚本帮助
python build_bm25_index.py --help
```
