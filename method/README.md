# method/ 代码定位方法框架

本目录实现了统一的代码定位评测框架，支持多种切块策略和检索方法。

## 快速开始

### 1. 激活 Conda 环境（必须）

```bash
# 如果使用 tmux 或新终端，需要先激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh  # 如果 conda 命令不可用
conda activate locagent
```

### 2. 安装依赖（如缺失）

```bash
# 如果遇到 ModuleNotFoundError，安装依赖
pip install transformers torch tqdm
# 或安装完整依赖
pip install -r requirements.txt
```

### 3. 设置环境变量

```bash
cd /workspace/LocAgent
export PYTHONPATH=/workspace/LocAgent
export TOKENIZERS_PARALLELISM=false
```

## 可用方法

| 方法 | 入口脚本 | 说明 |
|------|----------|------|
| Dense | `method/dense/run.py` | 稠密向量检索（RLRetriever） |
| Jaccard | `method/jaccard/run.py` | BoW 词袋检索（无需模型） |
| BM25 | `method/bm25/run.py` | BM25 检索（需预建索引） |

## 检索命令

### Dense 检索（使用预建索引，推荐）

**支持文件/模块/函数三级定位：**
```bash
python method/dense/run_with_index.py \
    --index_dir index_data/dense_index_fixed \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/dense_locator_fixed \
    --model_name models/rlretriever \
    --repos_root playground/locbench_repos \
    --top_k_blocks 50 \
    --top_k_files 10 \
    --top_k_modules 10 \
    --top_k_entities 50
```

**参数说明：**
- `--repos_root`: 代码仓库根目录（必需，用于AST映射器）
- `--top_k_modules`: 返回的模块数量（默认10）
- `--top_k_entities`: 返回的实体数量（默认50）
- **映射方式**: AST解析映射器（运行时解析源代码）

### Dense 检索（运行时动态切块，无需索引）

```bash
python method/RepoCoder/run_locator.py \
    --repos_root playground/locbench_repos \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/dense_locator \
    --model_name models/rlretriever \
    --mode dense \
    --block_size 15 \
    --top_k_blocks 50 --top_k_files 10
```

### Jaccard 检索（无需模型）

```bash
python method/RepoCoder/run_locator.py \
    --repos_root playground/locbench_repos \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/jaccard_locator \
    --mode jaccard \
    --block_size 15 \
    --top_k_blocks 50 --top_k_files 10
```

### BM25 检索（使用Graph映射器）

**支持文件/模块/函数三级定位：**
```bash
python method/bm25/run.py \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/bm25_results \
    --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \
    --bm25_index_dir index_data/Loc-Bench_V1/BM25_index \
    --top_k_files 10 \
    --top_k_modules 10 \
    --top_k_entities 50
```

**参数说明：**
- `--graph_index_dir`: Graph索引目录（必需，用于Graph映射器）
- `--bm25_index_dir`: BM25索引目录（必需）
- `--top_k_modules`: 返回的模块数量（默认10）
- `--top_k_entities`: 返回的实体数量（默认10）
- **映射方式**: Graph索引+span_ids映射器（依赖预构建的Graph索引）

## 索引构建（4种切块策略）

```bash
# 固定行切块（15行）- 全量仓库
python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever \
    --strategy fixed --block_size 15 \
    --max_length 512 --batch_size 8 \
    --num_processes 1

# 滑动窗口切块
python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever \
    --strategy sliding --window_size 20 --slice_size 2 \
    --num_processes 1

# RLCoder 固定块（12非空行）
python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever \
    --strategy rl_fixed \
    --num_processes 1

# RLCoder mini块
python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever \
    --strategy rl_mini \
    --num_processes 1
```

> 直接使用本地仓库，不需要网络。已处理的仓库会自动跳过。

## 评测结果

```bash
python -c "
from evaluation.eval_metric import evaluate_results
level2key = {'file':'found_files','module':'found_modules','function':'found_entities'}
print(evaluate_results('outputs/dense_locator/loc_outputs.jsonl',
                       level2key,
                       dataset_path='data/Loc-Bench_V1_dataset.jsonl'))
"
```

## 详细文档

完整命令和配置说明请参考 [FRAMEWORK.md](./FRAMEWORK.md)。
