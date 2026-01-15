# LocBench数据集测评运行逻辑详解

## 概述

LocBench数据集的测评流程主要分为三个阶段：
1. **索引构建阶段**：为代码仓库构建稠密向量索引
2. **检索运行阶段**：使用索引进行代码定位检索
3. **评估计算阶段**：计算各种评估指标

---

## 第一阶段：索引构建（Index Building）

### 核心脚本
- `method/index/batch_build_index.py`

### 运行流程

#### 1.1 输入准备
- **仓库路径**：`--repo_path /workspace/locbench/repos/locbench_repos`
  - 包含所有需要索引的代码仓库
- **切块策略**：`--strategy` 参数指定
  - `ir_function`: IR-based论文方法（函数级，模块级代码冗余复制）
  - `llamaindex_code`: LlamaIndex CodeSplitter（基于AST的代码切块）
  - `function_level`: AST提取函数和方法
  - `fixed`: 固定行数切块
  - 其他策略...

#### 1.2 代码切块（Chunking）

根据策略不同，代码切块方式不同：

**IR-based方法（ir_function策略）**：
```python
# 核心逻辑在 blocks_ir_function() 函数
1. 解析Python文件的AST
2. 提取模块级代码（imports, 全局变量）
3. 提取类属性
4. 对每个函数/方法：
   - 构建表示：file_path + class_name + module_code + class_attrs + function_code
   - 模块级代码和类属性会冗余复制到每个函数
5. 只处理Python文件中的函数
```

**LlamaIndex Code方法（llamaindex_code策略）**：
```python
# 核心逻辑在 collect_llamaindex_code_blocks() 函数
1. 使用 SimpleDirectoryReader 读取仓库文件
2. 使用 CodeSplitter 进行代码切块：
   - chunk_lines: 每个块的代码行数（默认40）
   - chunk_lines_overlap: 块之间的重叠行数（默认15）
   - max_chars: 每个块的最大字符数（默认1500）
3. 转换为Block对象，包含文件路径、行号范围、内容
```

#### 1.3 向量化（Embedding）

```python
# 核心逻辑在 embed_blocks() 函数
1. 为每个代码块构建输入文本：
   "file path: {file_path}\nlines: {start}-{end}\n\n{content}"
2. 使用预训练模型（如CodeRankEmbed）进行编码
3. 对token embeddings进行池化（mean pooling）
4. L2归一化
5. 批量处理，支持GPU加速
```

#### 1.4 索引保存

```python
# 核心逻辑在 save_index() 函数
每个仓库的索引保存为：
- embeddings.pt: 向量矩阵 (num_blocks, embedding_dim)
- metadata.jsonl: 每个块的元数据
  {
    "block_id": 0,
    "file_path": "src/utils.py",
    "start_line": 10,
    "end_line": 25,
    "block_type": "ir_function",
    "strategy": "ir_function",
    "qualified_name": "src/utils.py::MathUtils::calculate_sum",  # 函数级策略特有
    ...
  }
```

#### 1.5 多进程并行

```python
# 使用 torch.multiprocessing 进行多进程并行
1. 每个进程分配一个GPU（通过CUDA_VISIBLE_DEVICES）
2. 从共享队列获取仓库任务
3. 每个进程独立加载模型、处理仓库、保存索引
4. 支持进度监控和错误处理
```

### 示例命令

```bash
python method/index/batch_build_index.py \
    --repo_path /workspace/locbench/repos/locbench_repos \
    --strategy ir_function \
    --index_dir new_index_data/ir_function_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 4 \
    --gpu_ids 4,5,6,7 \
    --batch_size 32
```

---

## 第二阶段：检索运行（Retrieval）

### 核心脚本
- `method/dense/run_with_index.py`

### 运行流程

#### 2.1 加载模型和数据集

```python
1. 加载编码模型（用于编码查询文本）
2. 加载LocBench数据集（JSONL格式）
3. 提取每个实例的problem_statement作为查询文本
```

#### 2.2 编码查询

```python
# 核心逻辑在 embed_texts() 函数
1. 批量编码所有查询文本
2. 使用与索引构建相同的模型和编码方式
3. 输出查询向量 (num_queries, embedding_dim)
```

#### 2.3 检索过程

对每个实例：

```python
1. 根据instance_id确定仓库名称
   instance_id: "django__django-12345"
   repo_name: "django__django"

2. 加载对应仓库的索引
   - embeddings.pt: 代码块向量
   - metadata.jsonl: 代码块元数据

3. 计算相似度
   scores = query_embedding @ embeddings.T  # 余弦相似度（已归一化）

4. 获取Top-K代码块
   topk_blocks = topk(scores, k=top_k_blocks)

5. 聚合到文件级别
   - 对每个文件，累加其包含的所有代码块的分数
   - 按分数排序，取Top-K文件
```

#### 2.4 映射到模块和函数

```python
# 核心逻辑在 mapper.map_blocks_to_entities()

根据mapper_type选择映射器：

AST映射器（mapper_type="ast"）：
1. 解析仓库的AST
2. 根据代码块的行号范围，找到对应的：
   - 模块（module）：类定义
   - 实体（entity）：函数/方法定义
3. 返回Top-K模块和实体

Graph映射器（mapper_type="graph"）：
1. 使用Graph索引中的span_ids
2. 直接映射到对应的模块和实体
3. 更准确但需要额外的Graph索引
```

#### 2.5 输出结果

```json
{
  "instance_id": "django__django-12345",
  "found_files": ["django/core/management/commands/runserver.py", ...],
  "found_modules": ["django/core/management/commands/runserver.py:Command", ...],
  "found_entities": ["django/core/management/commands/runserver.py:Command.handle", ...],
  "raw_output_loc": []
}
```

### 示例命令

```bash
python method/dense/run_with_index.py \
  --index_dir new_index_data/llamacode_CodeRankEmbed/dense_index_llamaindex_code \
  --dataset_path /workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --output_folder new_outputs/llamacode_CodeRankEmbed \
  --model_name models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --repos_root /workspace/locbench/repos/locbench_repos \
  --gpu_id 1 \
  --batch_size 16 \
  --top_k_blocks 50 \
  --top_k_files 10
```

---

## 第三阶段：评估计算（Evaluation）

### 核心脚本
- `evaluation/eval_metric.py`
- `evaluation/run_evaluation.ipynb`（Jupyter notebook）

### 评估指标

支持多种评估指标：
- **Acc@K**: 准确率（前K个预测中是否包含所有真实位置）
- **Recall@K**: 召回率（前K个预测中包含的真实位置比例）
- **NDCG@K**: 归一化折损累积增益
- **Precision@K**: 精确率
- **MAP@K**: 平均精确率均值

### 评估级别

在三个粒度级别进行评估：

1. **文件级别（file）**
   - 从`found_files`字段读取预测
   - 从GT的`file_changes[].file`读取真实值
   - K值：通常为 [1, 3, 5]

2. **模块级别（module）**
   - 从`found_modules`字段读取预测
   - 从GT的`file_changes[].changes.edited_modules`读取真实值
   - K值：通常为 [5, 10]

3. **函数级别（function）**
   - 从`found_entities`字段读取预测
   - 从GT的`file_changes[].changes.edited_entities`读取真实值
   - K值：通常为 [5, 10]

### 评估流程

```python
# 核心逻辑在 evaluate_results() 函数

1. 加载定位结果文件（loc_outputs.jsonl）
2. 加载GT文件（Loc-Bench_V1_dataset.jsonl）
3. 对每个级别（file/module/function）：
   a. 提取预测位置列表
   b. 提取真实位置列表
   c. 构建标签向量（binary）
   d. 计算各种指标
4. 合并结果，输出DataFrame
```

### 标签构建逻辑

```python
# 对每个实例：
gt_labels = [0] * max_k  # 初始化全0
pred_labels = [0] * max_k  # 初始化全0

# 真实位置标记为1（前max_k个）
for i in range(len(gt_locations)):
    if i < max_k:
        gt_labels[i] = 1

# 预测位置如果在真实位置中，标记为1
for i, pred_loc in enumerate(pred_locations[:max_k]):
    if pred_loc in gt_locations:
        pred_labels[i] = 1
```

### 示例代码

```python
from evaluation.eval_metric import evaluate_results

level2key_dict = {
    'file': 'found_files',
    'module': 'found_modules',
    'function': 'found_entities',
}

loc_file = '/workspace/locbench/IR-based/new_outputs/ir_function_CodeRankEmbed/loc_outputs.jsonl'
dataset_path = '/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl'

result = evaluate_results(
    loc_file,
    level2key_dict,
    metrics=['acc', 'recall', 'ndcg', 'precision', 'map'],
    dataset_path=dataset_path
)
```

---

## 完整运行示例

### 步骤1：构建索引

```bash
# IR-based方法
python method/index/batch_build_index.py \
    --repo_path /workspace/locbench/repos/locbench_repos \
    --strategy ir_function \
    --index_dir new_index_data/ir_function_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 4 \
    --gpu_ids 4,5,6,7 \
    --batch_size 32
```

### 步骤2：运行检索

```bash
python method/dense/run_with_index.py \
  --index_dir new_index_data/ir_function_CodeRankEmbed/dense_index_ir_function \
  --dataset_path /workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --output_folder new_outputs/ir_function_CodeRankEmbed \
  --model_name models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --repos_root /workspace/locbench/repos/locbench_repos \
  --gpu_id 1 \
  --batch_size 16 \
  --top_k_blocks 50 \
  --top_k_files 10
```

### 步骤3：评估结果

在Jupyter notebook中运行：

```python
import sys
sys.path.append('/workspace/locbench/IR-based')

from evaluation.eval_metric import evaluate_results

level2key_dict = {
    'file': 'found_files',
    'module': 'found_modules',
    'function': 'found_entities',
}

loc_file = '/workspace/locbench/IR-based/new_outputs/ir_function_CodeRankEmbed/loc_outputs.jsonl'
dataset_path = '/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl'

result = evaluate_results(
    loc_file,
    level2key_dict,
    metrics=['acc', 'recall', 'ndcg', 'precision', 'map'],
    dataset_path=dataset_path
)

print(result)
```

---

## 关键技术细节

### 1. 文件路径规范化

```python
# 在 clean_file_path() 函数中
- 统一使用相对路径
- 处理不同格式的路径分隔符
- 确保与GT格式一致
```

### 2. 函数名处理

```python
# 处理 __init__ 方法
- 如果函数名以 ".__init__" 结尾，去除该后缀
- 确保与GT格式一致
```

### 3. 索引缓存

```python
# 在检索阶段使用索引缓存
- 避免重复加载同一仓库的索引
- 索引保留在CPU，只在计算时临时移到GPU
- 节省GPU内存
```

### 4. 多进程GPU分配

```python
# 在索引构建阶段
- 每个进程通过CUDA_VISIBLE_DEVICES限制可见GPU
- 避免多进程同时初始化所有GPU导致的死锁
- 支持动态GPU分配
```

---

## 常见问题

### Q1: 索引构建失败怎么办？
- 检查仓库路径是否正确
- 检查模型路径是否存在
- 检查GPU内存是否充足
- 查看日志中的具体错误信息

### Q2: 检索结果为空？
- 检查索引目录是否正确
- 检查仓库名称映射是否正确（instance_id -> repo_name）
- 检查索引文件是否存在（embeddings.pt, metadata.jsonl）

### Q3: 评估指标异常？
- 检查输出文件格式是否正确
- 检查GT文件路径是否正确
- 检查level2key_dict映射是否正确

---

## 总结

LocBench数据集的测评流程是一个完整的检索-评估pipeline：

1. **索引构建**：将代码仓库切块、向量化、保存索引
2. **检索运行**：使用索引进行相似度检索，映射到文件/模块/函数
3. **评估计算**：计算多粒度、多指标的评估结果

整个流程支持多种切块策略、多种评估指标，可以灵活配置和扩展。
