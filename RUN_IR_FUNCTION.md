# ir_function 索引构建和测评完整命令

## 1. 索引构建（Index Building）

### 完整命令

```bash
cd /workspace/locbench/IR-based

python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --strategy ir_function \
    --index_dir index_data/ir_function_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 4 \
    --gpu_ids 4,5,6,7 \
    --batch_size 32 \
    --max_length 512
```

### 参数说明

- `--repo_path`: 代码仓库根目录
- `--strategy`: 切块策略，使用 `ir_function`（函数级索引）
- `--index_dir`: 索引输出目录
- `--model_name`: 模型路径（如 `models/CodeRankEmbed`，或 HuggingFace 模型 ID 如 `LLukas22/CodeRankEmbed`）
- `--trust_remote_code`: CodeRankEmbed 模型需要此参数
- `--num_processes`: 并行进程数（建议与 GPU 数量一致）
- `--gpu_ids`: 指定使用的 GPU ID（逗号分隔）
- `--batch_size`: 批量大小（可根据 GPU 内存调整）
- `--max_length`: 最大 token 长度（默认 512）

### 可选的额外参数

如果需要从数据集获取仓库列表（而不是扫描本地目录）：

```bash
python method/index/batch_build_index.py \
    --dataset czlll/Loc-Bench_V1 \
    --split test \
    --strategy ir_function \
    --index_dir index_data/ir_function_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 4 \
    --gpu_ids 4,5,6,7 \
    --batch_size 32 \
    --max_length 512
```

---

## 2. Dense 检索（Retrieval）

索引构建完成后，使用预建的索引进行检索：

```bash
cd /workspace/locbench/IR-based

# 注意：model_name 必须与构建索引时使用的模型路径一致
python method/dense/run_with_index.py \
    --index_dir index_data/ir_function_CodeRankEmbed \
    --dataset_path ../data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/ir_function_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --repos_root playground/locbench_repos \
    --top_k_blocks 50 \
    --top_k_files 15 \
    --top_k_modules 15 \
    --top_k_entities 50 \
    --gpu_id 0 \
    --batch_size 8 \
    --max_length 512
```

### 参数说明

- `--index_dir`: 索引目录（与构建索引时的 `--index_dir` 一致）
- `--dataset_path`: 数据集 JSONL 文件路径
- `--output_folder`: 输出目录（检索结果保存在此）
- `--model_name`: 模型路径（必须与构建索引时使用的模型一致）
- `--trust_remote_code`: CodeRankEmbed 模型需要此参数
- `--repos_root`: 代码仓库根目录（用于 AST 映射器）
- `--top_k_blocks`: Top-K 代码块数量
- `--top_k_files`: Top-K 文件数量
- `--top_k_modules`: Top-K 模块数量
- `--top_k_entities`: Top-K 实体（函数）数量
- `--gpu_id`: 使用的 GPU ID（检索阶段通常只需要一个 GPU）
- `--batch_size`: 批量大小（查询编码）
- `--max_length`: 最大 token 长度（必须与构建索引时一致）

### 测试模式（运行前 N 个实例）

```bash
python method/dense/run_with_index.py \
    --index_dir index_data/ir_function_CodeRankEmbed \
    --dataset_path ../data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/ir_function_CodeRankEmbed_test \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --repos_root playground/locbench_repos \
    --top_k_blocks 50 \
    --top_k_files 15 \
    --top_k_modules 15 \
    --top_k_entities 50 \
    --eval_n_limit 10 \
    --gpu_id 0 \
    --batch_size 8 \
    --max_length 512
```

---

## 3. 评估结果（Evaluation）

检索完成后，评估结果：

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
    'outputs/ir_function_CodeRankEmbed/loc_outputs.jsonl',
    level2key,
    dataset_path='../data/Loc-Bench_V1_dataset.jsonl'
)
print(results)
"
```

### 保存评估结果到文件

```bash
cd /workspace/locbench/IR-based

python -c "
import sys
import pandas as pd
sys.path.insert(0, '../LocAgent')
from evaluation.eval_metric import evaluate_results

level2key = {
    'file': 'found_files',
    'module': 'found_modules',
    'function': 'found_entities'
}

results = evaluate_results(
    'outputs/ir_function_CodeRankEmbed/loc_outputs.jsonl',
    level2key,
    dataset_path='../data/Loc-Bench_V1_dataset.jsonl'
)

# 保存结果
output_file = 'outputs/ir_function_CodeRankEmbed/evaluation_results.csv'
results.to_csv(output_file)
print(f'评估结果已保存到: {output_file}')
print(results)
"
```

---

## 4. 完整流程（一键运行）

### 步骤 1: 构建索引

```bash
cd /workspace/locbench/IR-based

python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --strategy ir_function \
    --index_dir index_data/ir_function_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 4 \
    --gpu_ids 4,5,6,7 \
    --batch_size 32 \
    --max_length 512
```

**预计时间**：根据仓库数量和 GPU 性能，可能需要数小时。

### 步骤 2: 运行检索

```bash
cd /workspace/locbench/IR-based

# 注意：model_name 必须与构建索引时使用的模型路径一致
python method/dense/run_with_index.py \
    --index_dir index_data/ir_function_CodeRankEmbed \
    --dataset_path ../data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/ir_function_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --repos_root playground/locbench_repos \
    --top_k_blocks 50 \
    --top_k_files 15 \
    --top_k_modules 15 \
    --top_k_entities 50 \
    --gpu_id 0 \
    --batch_size 8 \
    --max_length 512
```

**预计时间**：根据数据集大小，可能需要 1-2 小时。

### 步骤 3: 评估结果

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
    'outputs/ir_function_CodeRankEmbed/loc_outputs.jsonl',
    level2key,
    dataset_path='../data/Loc-Bench_V1_dataset.jsonl'
)
print(results)
"
```

---

## 5. 注意事项

1. **模型路径**：
   - 确保 `models/CodeRankEmbed` 存在（已在 IR-based 目录下）
   - 如果模型在 HuggingFace Hub 上，可以使用模型 ID（如 `LLukas22/CodeRankEmbed`）

2. **索引目录**：
   - 索引会保存到 `index_data/ir_function_CodeRankEmbed/` 目录
   - 每个仓库会有独立的子目录，包含 `embeddings.pt` 和 `metadata.jsonl` 文件

3. **GPU 使用**：
   - 索引构建阶段：使用多个 GPU 并行处理（`--gpu_ids 4,5,6,7`）
   - 检索阶段：通常只需要一个 GPU（`--gpu_id 0`）

4. **参数一致性**：
   - 构建索引和检索时必须使用**相同的模型**和 **max_length**
   - 否则 embedding 维度不匹配，会导致错误

5. **输出文件**：
   - 检索结果保存在 `outputs/ir_function_CodeRankEmbed/loc_outputs.jsonl`
   - 格式：每行一个 JSON 对象，包含 `instance_id`, `found_files`, `found_modules`, `found_entities`

6. **评估指标**：
   - 默认评估：Acc@K, Recall@K, Precision@K, NDCG@K, MAP@K
   - K 值：file [1,3,5], module [5,10], function [5,10]

---

## 6. 故障排查

### 问题：ModuleNotFoundError

如果遇到模块导入错误，确保设置了正确的 Python 路径：

```bash
export PYTHONPATH="/workspace/locbench/IR-based:/workspace/locbench/LocAgent:$PYTHONPATH"
```

### 问题：GPU 内存不足

如果 GPU 内存不足，可以：
- 减小 `--batch_size`（如 16 或 8）
- 减小 `--num_processes`（减少并行数）
- 使用 `--force_cpu`（不推荐，速度很慢）

### 问题：索引已存在

如果索引目录已存在，脚本会跳过已处理的仓库。如果要重新构建，先删除索引目录：

```bash
rm -rf index_data/ir_function_CodeRankEmbed
```

### 问题：检索时维度不匹配

确保检索时使用的模型和 `max_length` 与构建索引时一致。
