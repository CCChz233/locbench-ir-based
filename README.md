# IR-based: 基于索引的代码定位方法

本目录包含基于索引检索的代码定位方法实现，从 LocAgent 中分离出来，专注于纯检索方法。

## 目录结构

```
IR-based/
├── method/              # 各种检索方法实现
│   ├── bm25/           # BM25 检索方法
│   ├── dense/          # 稠密向量检索方法
│   ├── jaccard/        # Jaccard 相似度检索
│   ├── mapping/        # 代码块到实体的映射器
│   ├── index/          # 索引构建工具
│   ├── RepoCoder/      # RepoCoder 方法
│   ├── RLCoder/        # RLCoder 方法
│   ├── base.py         # 基础类和数据结构
│   └── utils.py        # 工具函数
├── index_data/         # 索引文件（符号链接到 LocAgent/index_data/）
├── scripts/            # 运行脚本
│   ├── run_bm25_baseline.py
│   └── run_retrieval_benchmark.py
├── build_bm25_index.py # BM25 索引构建脚本
└── README.md           # 本文件
```

## 依赖关系

本框架依赖 LocAgent 的以下共享模块（通过 sys.path 自动访问）：

- `LocAgent/evaluation/`: 评估模块
- `LocAgent/dependency_graph/`: 图构建模块
- `LocAgent/util/`: 工具函数库
- `LocAgent/plugins/`: 插件系统

索引数据通过符号链接共享：`index_data/` → `../LocAgent/index_data/`

## 环境设置

### 1. 激活 Conda 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh  # 如果需要
conda activate locagent
```

### 2. 设置环境变量

```bash
cd /workspace/locbench/IR-based
export PYTHONPATH="/workspace/locbench/IR-based:/workspace/locbench/LocAgent:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
```

## 使用方法

### 运行 BM25 基线

```bash
python scripts/run_bm25_baseline.py \
    --dataset_path ../data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/bm25_results \
    --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \
    --bm25_index_dir index_data/Loc-Bench_V1/BM25_index \
    --top_k_files 10 \
    --top_k_modules 10 \
    --top_k_entities 50
```

### 构建 BM25 索引

```bash
python build_bm25_index.py \
    --dataset czlll/Loc-Bench_V1 \
    --split test \
    --num_processes 4 \
    --repo_path ../repos/locbench_repos \
    --index_dir index_data
```

### 运行 Dense 检索

```bash
python method/dense/run_with_index.py \
    --index_dir index_data/dense_index_fixed \
    --dataset_path ../data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/dense_results \
    --model_name models/rlretriever \
    --repos_root ../repos/locbench_repos \
    --top_k_files 10 \
    --top_k_modules 10 \
    --top_k_entities 50
```

### 运行 Jaccard 检索

```bash
python method/RepoCoder/run_locator.py \
    --repos_root ../repos/locbench_repos \
    --dataset_path ../data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/jaccard_results \
    --mode jaccard \
    --block_size 15 \
    --top_k_blocks 50 \
    --top_k_files 10
```

## 评估结果

运行评估：

```bash
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
    'outputs/bm25_results/loc_outputs.jsonl',
    level2key,
    dataset_path='../data/Loc-Bench_V1_dataset.jsonl'
)
print(results)
"
```

或者使用 Jupyter notebook：参考 `../LocAgent/evaluation/run_evaluation.ipynb`

## 可用方法

| 方法 | 入口脚本 | 说明 |
|------|----------|------|
| **BM25** | `scripts/run_bm25_baseline.py` | BM25 检索（需预建索引） |
| **Dense** | `method/dense/run_with_index.py` | 稠密向量检索（RLRetriever） |
| **Jaccard** | `method/RepoCoder/run_locator.py` | BoW 词袋检索（无需模型） |

## 索引构建

### BM25 索引

```bash
python build_bm25_index.py \
    --dataset czlll/Loc-Bench_V1 \
    --split test \
    --num_processes 4 \
    --repo_path ../repos/locbench_repos \
    --index_dir index_data
```

### Dense 索引（4种切块策略）

```bash
# 固定行切块（15行）
python method/index/batch_build_index.py \
    --repo_path ../repos/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever \
    --strategy fixed \
    --block_size 15 \
    --max_length 512 \
    --batch_size 8 \
    --num_processes 1

# 滑动窗口切块
python method/index/batch_build_index.py \
    --repo_path ../repos/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever \
    --strategy sliding \
    --window_size 20 \
    --slice_size 2 \
    --max_length 512 \
    --batch_size 8 \
    --num_processes 1
```

更多索引构建选项请参考 `method/README.md` 和 `method/FRAMEWORK.md`。

## 与 LocAgent 的关系

- **IR-based**: 纯索引检索方法，无交互式 Agent，专注于基于索引的代码定位
- **LocAgent**: 交互式 LLM Agent 方法，使用图引导的多轮搜索

两者共享以下基础设施：
- 评估模块（`LocAgent/evaluation/`）
- 图构建模块（`LocAgent/dependency_graph/`）
- 工具函数库（`LocAgent/util/`）
- 插件系统（`LocAgent/plugins/`）
- 索引数据（通过符号链接共享）

## 路径说明

- 数据集：`../data/Loc-Bench_V1_dataset.jsonl`
- 代码仓库：`../repos/locbench_repos/`
- 索引数据：`index_data/`（符号链接到 `../LocAgent/index_data/`）
- 模型文件：`models/`（已在 IR-based 目录下）
- 输出目录：`outputs/`

## 注意事项

1. **路径处理**：脚本已自动处理路径，无需手动设置 PYTHONPATH（但建议设置以确保一致性）
2. **索引数据**：索引数据通过符号链接共享，如需独立副本可手动复制
3. **相对路径**：脚本中的相对路径基于 `IR-based/` 目录
4. **环境依赖**：需要 LocAgent 的 conda 环境和依赖包

## 故障排查

### 导入错误

如果遇到 `ModuleNotFoundError`，确保：
1. PYTHONPATH 包含 IR-based 和 LocAgent 目录
2. LocAgent 目录结构完整
3. Conda 环境已激活

### 索引文件找不到

检查索引路径是否正确：
- 默认路径：`index_data/{dataset_name}/graph_index_v2.3/`
- 可通过环境变量或命令行参数指定：`--graph_index_dir`, `--bm25_index_dir`

## 更多信息

- 详细方法说明：`method/README.md`
- 框架文档：`method/FRAMEWORK.md`
- LocAgent 文档：`../LocAgent/README.md`
