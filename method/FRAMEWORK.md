# 方法评测框架

## 核心目标
为不同代码定位算法提供统一的评测框架，便于横向对比和复现。

## 服务器环境配置

### 1. 激活 Conda 环境（必须）

```bash
# 如果使用 tmux 或新终端，需要先激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh  # 如果 conda 命令不可用
conda activate locagent

# 验证环境
which python  # 应该显示 /root/miniconda3/envs/locagent/bin/python
python -c "import transformers; print('✓ transformers 已安装')"
```

### 2. 安装依赖（如缺失）

```bash
# 如果遇到 ModuleNotFoundError: No module named 'transformers'
pip install transformers torch tqdm

# 或者安装完整依赖（推荐）
pip install -r requirements.txt
```

### 3. 环境变量设置

```bash
# 工作目录
cd /workspace/LocAgent
export PYTHONPATH=/workspace/LocAgent
export TOKENIZERS_PARALLELISM=false
```

### 路径约定
| 路径 | 说明 |
|------|------|
| `/workspace/LocAgent` | 项目根目录 |
| `playground/locbench_repos` | 代码仓库目录（97个仓库） |
| `outputs` | 输出结果目录 |
| `data/Loc-Bench_V1_dataset.jsonl` | 数据集 |
| `models/rlretriever` | RLRetriever 模型 |
| `index_data` | 索引存储目录 |

---

## 三层结构
- **统一接口层**（`method/base.py`）：定义标准输出 `LocResult` 与可选基类 `BaseMethod`。
- **共享工具层**（`method/utils.py`）：数据集加载、结果保存、索引加载、通用参数解析。
- **方法实现层**（`method/{method_name}/`）：每个方法独立目录，包含 `run.py` 入口和索引构建脚本。
- **映射模块**（`method/mapping/`）：代码块到函数/模块的映射实现
  - `graph_based/`: Graph索引+span_ids映射（用于BM25）
  - `ast_based/`: AST解析映射（用于Dense，运行时解析）

---

## 映射方式

代码块到函数/模块的映射支持两种方式：

| 映射方式 | 实现位置 | 使用场景 | 依赖 |
|---------|---------|---------|------|
| **Graph索引+span_ids** | `method/mapping/graph_based/` | BM25检索 | Graph索引（预构建） |
| **AST解析** | `method/mapping/ast_based/` | Dense检索 | 源代码文件（运行时解析） |

**特点：**
- **Graph映射器**：使用预构建的Graph索引和代码块的`span_ids`，速度快但需要Graph索引
- **AST映射器**：运行时解析源代码文件，不依赖Graph索引，更公平但需要源代码

---

## 切块策略

统一索引构建器 `method/index/build_index.py` 支持4种切块策略：

| 策略 | 说明 | 参数 |
|------|------|------|
| `fixed` | 固定行切块（不重叠） | `--block_size 15` |
| `sliding` | 滑动窗口切块（可重叠） | `--window_size 20 --slice_size 2` |
| `rl_fixed` | RLCoder 固定块（12非空行） | 无额外参数 |
| `rl_mini` | RLCoder mini块（空行分段，拼接≤15行） | 无额外参数 |

---

## 🚀 可直接运行的命令

> 以下所有命令均在 `/workspace/LocAgent` 目录下运行。

### 环境初始化（每次新终端必须执行）

```bash
cd /workspace/LocAgent
export PYTHONPATH=/workspace/LocAgent
export TOKENIZERS_PARALLELISM=false
```

---

### 1. 索引构建

#### 1.1 全量稠密索引构建（多进程并行，推荐）

使用 `batch_build_index.py`（仿照 `batch_build_graph.py` 的多进程模式）：

```bash
# 固定行切块（15行一块）- 使用 GPU 4，全量仓库
python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever \
    --strategy fixed \
    --block_size 15 \
    --max_length 512 --batch_size 8 \
    --gpu_ids 4 \
    --num_processes 1

# 多卡并行（使用 GPU 4,5,6,7）
python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever \
    --strategy fixed \
    --block_size 15 \
    --max_length 512 --batch_size 8 \
    --gpu_ids 4,5,6,7 \
    --num_processes 4

# 滑动窗口切块 - 全量
python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever \
    --strategy sliding \
    --window_size 20 --slice_size 2 \
    --max_length 512 --batch_size 8 \
    --num_processes 1

# RLCoder 固定块（12非空行）- 全量
python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever \
    --strategy rl_fixed \
    --max_length 512 --batch_size 8 \
    --num_processes 1

# RLCoder mini块 - 全量
python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever \
    --strategy rl_mini \
    --max_length 512 --batch_size 8 \
    --num_processes 1
```

> **说明**：
> - 直接使用本地仓库目录，不需要网络
> - 已处理的仓库会自动跳过（检查 `embeddings.pt` 是否存在）
> - `--num_processes` 建议设为 1（单 GPU）或与 GPU 数量一致

#### 1.2 单仓库索引构建（测试用）

```bash
# 固定行切块（15行一块）
python method/index/build_index.py \
    --repo_path playground/locbench_repos/UXARRAY_uxarray \
    --output_dir index_data/dense_fixed/UXARRAY_uxarray \
    --model_name models/rlretriever \
    --strategy fixed \
    --block_size 15 \
    --max_length 512 --batch_size 8
```

#### 1.2 BM25 索引构建

```bash
python build_bm25_index.py \
    --dataset czlll/Loc-Bench_V1 \
    --split test \
    --repo_path playground/locbench_repos \
    --index_dir index_data \
    --num_processes 4
```
输出到 `index_data/Loc-Bench_V1/BM25_index/`。

#### 1.3 图索引构建

```bash
bash scripts/gen_graph_index.sh
```
输出到 `index_data/Loc-Bench_V1/graph_index_v2.3/`。

---

### 2. 检索运行

#### 2.1 Dense 检索（使用预建索引，推荐）

如果你已经构建了索引（使用 `batch_build_index.py`），可以使用预建索引进行检索：

**支持文件/模块/函数三级定位：**

```bash
# 使用 fixed 策略的索引
python method/dense/run_with_index.py \
    --index_dir index_data/dense_index_fixed \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/dense_locator_fixed \
    --model_name models/rlretriever \
    --repos_root playground/locbench_repos \
    --max_length 512 --batch_size 8 \
    --top_k_blocks 50 \
    --top_k_files 15 \
    --top_k_modules 15 \
    --top_k_entities 15

# 使用 sliding 策略的索引
python method/dense/run_with_index.py \
    --index_dir index_data/dense_index_sliding \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/dense_locator_sliding \
    --model_name models/rlretriever \
    --repos_root playground/locbench_repos \
    --top_k_blocks 50 \
    --top_k_files 15 \
    --top_k_modules 15 \
    --top_k_entities 15

# 使用 rl_fixed 策略的索引
python method/dense/run_with_index.py \
    --index_dir index_data/dense_index_rl_fixed \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/dense_locator_rl_fixed \
    --model_name models/rlretriever \
    --repos_root playground/locbench_repos \
    --top_k_blocks 50 \
    --top_k_files 15 \
    --top_k_modules 15 \
    --top_k_entities 15
```

**参数说明：**
- `--repos_root`: 代码仓库根目录（必需，用于AST映射器运行时解析源代码）
- `--top_k_modules`: 返回的模块数量（默认15）
- `--top_k_entities`: 返回的实体数量（默认15）
- **映射方式**: 使用 **AST解析映射器**（运行时解析，不依赖Graph索引）

> **优势**：速度快（无需运行时切块编码代码），适合批量实验。支持函数/模块级别定位。

#### 2.2 Dense 稠密检索（运行时动态切块，无需预建索引）

如果你没有预建索引，可以使用运行时动态切块（较慢）：

```bash
python method/RepoCoder/run_locator.py \
    --repos_root playground/locbench_repos \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/dense_locator \
    --model_name models/rlretriever \
    --mode dense \
    --block_size 15 \
    --max_length 512 --batch_size 8 \
    --top_k_blocks 50 --top_k_files 15
```

> **注意**：此方法需要运行时动态切块和编码代码，速度较慢，适合小规模测试。

#### 2.2 Dense 稠密检索（小样本冒烟测试）

```bash
python method/RepoCoder/run_locator.py \
    --repos_root playground/locbench_repos \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/dense_locator_smoke \
    --model_name models/rlretriever \
    --mode dense \
    --block_size 15 \
    --max_length 512 --batch_size 8 \
    --top_k_blocks 50 --top_k_files 15 \
    --eval_n_limit 10
```

#### 2.3 Jaccard/BoW 检索（无需模型）

```bash
python method/RepoCoder/run_locator.py \
    --repos_root playground/locbench_repos \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/jaccard_locator \
    --mode jaccard \
    --block_size 15 \
    --top_k_blocks 50 --top_k_files 15
```

#### 2.4 Jaccard 检索（小样本冒烟测试）

```bash
python method/RepoCoder/run_locator.py \
    --repos_root playground/locbench_repos \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/jaccard_locator_smoke \
    --mode jaccard \
    --block_size 15 \
    --top_k_blocks 50 --top_k_files 15 \
    --eval_n_limit 10
```

#### 2.5 BM25 检索（需预建索引）

**使用统一接口（推荐）：**
```bash
python method/bm25/run.py \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/bm25_results \
    --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \
    --bm25_index_dir index_data/Loc-Bench_V1/BM25_index \
    --top_k_files 15 \
    --top_k_modules 15 \
    --top_k_entities 15
```

**或使用原始脚本：**
```bash
python scripts/run_bm25_baseline.py \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/bm25_locbench \
    --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \
    --bm25_index_dir index_data/Loc-Bench_V1/BM25_index \
    --top_k_files 15 \
    --top_k_modules 15 \
    --top_k_entities 15
```

**参数说明：**
- `--graph_index_dir`: Graph索引目录（必需，用于Graph映射器）
- `--bm25_index_dir`: BM25索引目录（必需）
- `--top_k_modules`: 返回的模块数量（默认10）
- `--top_k_entities`: 返回的实体数量（默认10）
- 映射方式：使用 **Graph索引+span_ids映射器**（依赖预构建的Graph索引）

---

### 3. 简化入口脚本

```bash
# Dense 检索（使用预建索引，推荐）
python method/dense/run_with_index.py \
    --index_dir index_data/dense_index_fixed \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/dense_locator_fixed \
    --model_name models/rlretriever \
    --repos_root playground/locbench_repos \
    --top_k_blocks 50 \
    --top_k_files 15 \
    --top_k_modules 15 \
    --top_k_entities 15

# Dense 检索（运行时动态切块，无需索引）
python method/dense/run.py \
    --repos_root playground/locbench_repos \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/dense_locator \
    --model_name models/rlretriever \
    --block_size 15 \
    --top_k_blocks 50 --top_k_files 15

# Jaccard 检索（自动添加 --mode jaccard）
python method/jaccard/run.py \
    --repos_root playground/locbench_repos \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/jaccard_locator \
    --block_size 15 \
    --top_k_blocks 50 --top_k_files 15

# BM25 检索（使用Graph映射器）
python method/bm25/run.py \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/bm25_results \
    --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \
    --bm25_index_dir index_data/Loc-Bench_V1/BM25_index \
    --top_k_files 15 \
    --top_k_modules 15 \
    --top_k_entities 15
```

---

### 4. 结果评测

#### 4.1 评测单个结果文件

```bash
python -c "
from evaluation.eval_metric import evaluate_results
level2key = {'file':'found_files','module':'found_modules','function':'found_entities'}
print(evaluate_results('outputs/dense_locator/loc_outputs.jsonl',
                       level2key,
                       dataset_path='data/Loc-Bench_V1_dataset.jsonl'))
"
```

#### 4.2 评测 Jaccard 结果

```bash
python -c "
from evaluation.eval_metric import evaluate_results
level2key = {'file':'found_files','module':'found_modules','function':'found_entities'}
print(evaluate_results('outputs/jaccard_locator/loc_outputs.jsonl',
                       level2key,
                       dataset_path='data/Loc-Bench_V1_dataset.jsonl'))
"
```

#### 4.3 评测 BM25 结果

```bash
python -c "
from evaluation.eval_metric import evaluate_results
level2key = {'file':'found_files','module':'found_modules','function':'found_entities'}
print(evaluate_results('outputs/bm25_locbench/loc_outputs.jsonl',
                       level2key,
                       dataset_path='data/Loc-Bench_V1_dataset.jsonl'))
"
```

---

### 5. 快速冒烟测试

#### 5.1 索引构建冒烟（小目录测试）

```bash
python method/index/build_index.py \
    --repo_path dependency_graph \
    --output_dir index_data/smoke_test \
    --model_name models/rlretriever \
    --strategy fixed \
    --block_size 15 \
    --batch_size 4
```

#### 5.2 检索冒烟（限制5条）

```bash
python method/RepoCoder/run_locator.py \
    --repos_root playground/locbench_repos \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/smoke_test \
    --model_name models/rlretriever \
    --mode dense \
    --block_size 15 \
    --top_k_blocks 50 --top_k_files 15 \
    --eval_n_limit 5
```

---

## 输出格式

所有方法输出标准 `loc_outputs.jsonl`，每行格式：
```json
{
    "instance_id": "UXARRAY__uxarray-1117",
    "found_files": ["uxarray/grid/grid.py", "uxarray/core/dataarray.py"],
    "found_modules": ["uxarray/grid/grid.py:Grid", "uxarray/core/dataarray.py:UxDataArray"],
    "found_entities": ["uxarray/grid/grid.py:Grid.construct_face_centers", "uxarray/core/dataarray.py:UxDataArray.weighted_mean"],
    "raw_output_loc": []
}
```

**字段说明：**
- `found_files`: 文件路径列表（所有方法都支持）
- `found_modules`: 模块ID列表（格式：`file_path:ClassName` 或 `file_path:function_name`）
- `found_entities`: 实体ID列表（格式：`file_path:ClassName.method_name` 或 `file_path:function_name`）

**映射方式：**
- **BM25**: 使用Graph索引+span_ids映射器（依赖预构建的Graph索引）
- **Dense**: 使用AST解析映射器（运行时解析源代码，不依赖Graph索引）

---

## 测试重构后的代码

### 快速测试脚本

```bash
# 运行测试脚本（验证映射器功能）
python method/test_refactoring.py
```

测试内容：
- Graph映射器（需要Graph索引和BM25索引）
- AST映射器（需要源代码仓库）
- Dense集成检查

### 完整功能测试

**测试BM25（使用Graph映射器）：**
```bash
export GRAPH_INDEX_DIR="index_data/Loc-Bench_V1/graph_index_v2.3"
export BM25_INDEX_DIR="index_data/Loc-Bench_V1/BM25_index"

python method/bm25/run.py \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/test_bm25 \
    --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \
    --bm25_index_dir index_data/Loc-Bench_V1/BM25_index \
    --top_k_files 5 \
    --top_k_modules 5 \
    --top_k_entities 10 \
    --eval_n_limit 5
```

**测试Dense（使用AST映射器）：**
```bash
python method/dense/run_with_index.py \
    --index_dir index_data/dense_index_fixed \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/test_dense \
    --model_name models/rlretriever \
    --repos_root playground/locbench_repos \
    --top_k_blocks 50 \
    --top_k_files 15 \
    --top_k_modules 15 \
    --top_k_entities 15 \
    --eval_n_limit 5
```

**验证输出格式：**
```bash
python -c "
import json
with open('outputs/test_dense/loc_outputs.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        data = json.loads(line)
        print(f'Instance {data[\"instance_id\"]}:')
        print(f'  Files: {len(data[\"found_files\"])}, Modules: {len(data[\"found_modules\"])}, Entities: {len(data[\"found_entities\"])}')
        if data['found_modules']:
            print(f'  示例模块: {data[\"found_modules\"][0]}')
        if data['found_entities']:
            print(f'  示例实体: {data[\"found_entities\"][0]}')
"
```

> **注意**：重构后的代码不需要重构索引，映射是在运行时进行的。

---

## 目录结构

| 入口脚本 | 功能 |
|----------|------|
| `method/index/build_index.py` | 统一索引构建（4种策略） |
| `method/dense/build_index.py` | 固定行索引构建（包装） |
| `method/sliding/build_index.py` | 滑窗索引构建（包装） |
| `method/dense/run.py` | Dense检索 |
| `method/jaccard/run.py` | Jaccard检索 |
| `method/bm25/run.py` | BM25检索 |
| `method/RepoCoder/run_locator.py` | 底层定位器实现 |

---

## 模型下载（首次使用）

如果 `models/rlretriever` 不存在，需要下载：

```bash
huggingface-cli download nov3630/RLRetriever \
    --local-dir models/rlretriever \
    --resume-download \
    --local-dir-use-symlinks False
```

---

## 批量索引构建脚本示例

为所有仓库构建索引：

```bash
#!/bin/bash
cd /workspace/LocAgent
export PYTHONPATH=/workspace/LocAgent
export TOKENIZERS_PARALLELISM=false

STRATEGY="fixed"  # 可选: fixed, sliding, rl_fixed, rl_mini
OUTPUT_BASE="index_data/dense_${STRATEGY}"

for repo in playground/locbench_repos/*/; do
    repo_name=$(basename "$repo")
    echo "Building index for $repo_name..."
    python method/index/build_index.py \
        --repo_path "$repo" \
        --output_dir "${OUTPUT_BASE}/${repo_name}" \
        --model_name models/rlretriever \
        --strategy "$STRATEGY" \
        --block_size 15 \
        --max_length 512 --batch_size 8
done
```

---

## 常见问题

### Q: CUDA 内存不足怎么办？
A: 减小 `--batch_size`，或添加 `--force_cpu` 使用 CPU。

### Q: 如何限制测试数量？
A: 添加 `--eval_n_limit N` 只处理前 N 条。

### Q: 如何断点续跑？
A: 框架自动跳过已处理的 instance_id（检查输出文件）。
