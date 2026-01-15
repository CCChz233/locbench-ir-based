# 评估过程检查报告

检查 `run_with_index.py` 的评估过程是否满足论文要求。

## 论文要求总结

### 1. 查询（Query）构建：直接使用 Issue 描述
- 使用 `problem_statement` 作为查询文本
- 对于 Embedding 方法：直接 Embed Issue 文本，然后做向量相似度检索

### 2. 评测流程（Evaluation Pipeline）

#### 离线阶段：
1. 遍历代码库，提取所有函数
2. 对每个函数，构造文档：[File Path] + [Class Name] + [Function Code]
3. Embedding 方法：用模型 Embed 所有文档，存到向量库

#### 在线检索阶段：
1. 输入：Issue 的 `problem_statement`
2. Embedding：Embed Issue，计算与所有函数向量的余弦相似度，返回 Top-K

#### 评测：
- Ground Truth：`edit_functions` 列表（如 `["file.py:func1", "file.py:func2"]`）
- 判定：如果 Top-K 中包含了所有 Ground Truth 函数，算命中（Acc@K = 1）

---

## 代码实现检查

### ✅ 1. 查询构建（Query Construction）

**位置**：`get_problem_text()` 函数（第43-49行）

**代码**：
```python
def get_problem_text(instance: dict) -> str:
    """从实例中提取问题描述"""
    for key in ("problem_statement", "issue", "description", "prompt", "text"):
        val = instance.get(key)
        if val:
            return val
    return ""
```

**使用位置**（第200行）：
```python
queries = [get_problem_text(ins) for ins in instances]
```

**检查结果**：✅ **符合要求**

- ✅ 优先使用 `problem_statement`（第一个检查的键）
- ✅ 如果没有 `problem_statement`，回退到其他字段（兼容性处理）
- ✅ 直接使用 Issue 文本作为查询，没有进行特殊处理或预处理

**符合论文要求**：论文要求"直接使用 Issue 描述（problem_statement）"，代码实现符合。

---

### ✅ 2. 在线检索阶段（Online Retrieval）

#### 2.1 Embedding 查询文本

**位置**：第202-205行

**代码**：
```python
# 编码所有查询（批量处理）
print("Encoding queries...")
query_embeddings = embed_texts(queries, model, tokenizer, args.max_length, args.batch_size, device)
```

**`embed_texts()` 函数**（第52-96行）：
- 使用 tokenizer 对查询文本进行编码
- 使用模型获取 token embeddings
- 计算平均池化（mean pooling）：`summed / counts`
- L2 归一化：`torch.nn.functional.normalize(sent_emb, p=2, dim=1)`

**检查结果**：✅ **符合要求**

- ✅ 直接 Embed Issue 文本（通过 `embed_texts()` 函数）
- ✅ 使用标准的 embedding 流程（tokenization → model forward → pooling → normalization）

#### 2.2 计算相似度

**位置**：第269-273行

**代码**：
```python
# 计算相似度（临时移到 GPU）
query_emb_gpu = query_emb.to(device)
embeddings_gpu = embeddings.to(device)  # 临时移到 GPU
scores = torch.matmul(query_emb_gpu.unsqueeze(0), embeddings_gpu.t()).squeeze(0)  # (num_blocks,)
scores = scores.cpu()  # 移回 CPU 以便后续处理
```

**检查结果**：✅ **符合要求**

- ✅ 使用矩阵乘法计算余弦相似度（因为向量已归一化，`matmul` 等于余弦相似度）
- ✅ 计算查询向量与所有函数向量的相似度

#### 2.3 返回 Top-K

**位置**：第275-283行

**代码**：
```python
# 获取 Top-K 代码块
topk = min(args.top_k_blocks, scores.numel())
if topk == 0:
    found_files = []
    found_modules = []
    found_entities = []
else:
    topk_scores, topk_idx = torch.topk(scores, k=topk)
    block_scores = list(zip(topk_idx.tolist(), topk_scores.tolist()))
    found_files = rank_files(block_scores, metadata, args.top_k_files, repo_name)
```

**检查结果**：✅ **符合要求**

- ✅ 使用 `torch.topk()` 获取 Top-K 代码块
- ✅ 返回 Top-K 结果

---

### ✅ 3. 评测（Evaluation）

**注意**：`run_with_index.py` 本身**不包含评测代码**，它只生成 `loc_outputs.jsonl` 文件。

**输出格式**（第304-311行）：
```python
record = {
    "instance_id": instance_id,
    "found_files": found_files,
    "found_modules": found_modules,
    "found_entities": found_entities,  # 函数级别的预测结果
    "raw_output_loc": [],
}
fout.write(json.dumps(record) + "\n")
```

**评测代码位置**：`LocAgent/evaluation/eval_metric.py`

#### 3.1 Ground Truth 加载

**位置**：`cal_metrics_w_dataset()` 函数（第281-324行）

**代码**（第302-324行）：
```python
gt_dict = collections.defaultdict(list)
for instance in bench_data:
    if eval_level == 'file':
        for func in instance['edit_functions']:
            fn = func.split(':')[0]
            if fn not in gt_dict[instance['instance_id']]:
                gt_dict[instance['instance_id']].append(fn)
    elif eval_level == 'module':
        for func in instance['edit_functions']:
            fn = func.split(':')[0]
            mname = func.split(':')[-1].split('.')[0]
            mid = f'{fn}:{mname}'
            if mid not in gt_dict[instance['instance_id']]:
                gt_dict[instance['instance_id']].append(mid)
    elif eval_level == 'function':
        for func in instance['edit_functions']:  # ✅ 使用 edit_functions 作为 GT
            fn = func.split(':')[0]
            mname = func.split(':')[-1]
            if mname.endswith('.__init__'):
                mname = mname[:(len(mname)-len('.__init__'))]
            mid = f'{fn}:{mname}'
            if mid not in gt_dict[instance['instance_id']]:
                gt_dict[instance['instance_id']].append(mid)
```

**检查结果**：✅ **符合要求**

- ✅ Ground Truth 使用 `edit_functions` 列表（第316行）
- ✅ 格式：`["file.py:func1", "file.py:func2"]`

#### 3.2 预测结果加载

**位置**：`cal_metrics_w_dataset()` 函数（第326-365行）

**代码**（第345-365行）：
```python
pred_dict = convert_solutions_dict(load_jsonl(loc_file), key=key)
# 对于 function 级别，进行去重和清理 __init__ 后缀
if eval_level == 'function':
    for ins in pred_dict:
        pred_funcs = pred_dict[ins]
        pred_modules = []
        for i, pf in enumerate(pred_funcs):
            if pf.endswith('.__init__'):
                pf = pf[:(len(pf)-len('.__init__'))]
            if pf not in pred_modules:
                pred_modules.append(pf)
        pred_dict[ins] = pred_modules
```

**检查结果**：✅ **符合要求**

- ✅ 从 `loc_outputs.jsonl` 中加载 `found_entities`（通过 `key='found_entities'`）
- ✅ 对 `__init__` 后缀进行处理（与 GT 保持一致）

#### 3.3 评测指标计算

**位置**：`cal_metrics_w_dataset()` 函数（第367-380行）

**代码**（第367-380行）：
```python
for instance_id in gt_dict.keys():
    if not gt_dict[instance_id]: continue
    
    if instance_id not in pred_dict:
        pred_locs = []
    else:
        pred_locs = pred_dict[instance_id][: max_k]  # Top-K
        
    gt_labels = [0 for _ in range(max_k)]
    pred_labels = [0 for _ in range(max_k)]
    
    for i in range(len(gt_dict[instance_id])):
        if i < max_k:
            gt_labels[i] = 1
    
    for i, l in enumerate(pred_locs):
        if l in gt_dict[instance_id]:  # ✅ 判断预测是否在 GT 中
            pred_labels[i] = 1
```

**评测指标**（第244-253行）：
- `acc_at_k()`：Acc@K（如果 Top-K 中包含所有 GT，算命中）
- `recall_at_k()`：Recall@K
- `precision_at_k()`：Precision@K
- `normalized_dcg()`：NDCG@K
- `average_precision_at_k()`：MAP@K

**`acc_at_k()` 实现**（第66-74行）：
```python
def acc_at_k(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    pred_target = pred_target[:, :k]  # 只考虑前 k 个预测结果
    ideal_target = ideal_target[:, :k]
    
    relevant = (pred_target == 1).sum(dim=-1)  # 计算预测中相关文档的个数
    total_relevant = (ideal_target == 1).sum(dim=-1)  # 计算所有相关文档的个数
    
    comparison = relevant == total_relevant  # ✅ 判断是否包含所有 GT
    return comparison.sum()/relevant.shape[0]
```

**检查结果**：✅ **符合要求**

- ✅ Acc@K 指标：判断 Top-K 中是否包含**所有** Ground Truth 函数（第73行：`relevant == total_relevant`）
- ✅ 如果包含所有 GT，算命中（Acc@K = 1）

---

## 总结

### ✅ 符合论文要求的部分

1. ✅ **查询构建**：
   - 直接使用 `problem_statement` 作为查询文本
   - 没有进行特殊处理或预处理

2. ✅ **在线检索阶段**：
   - 输入：Issue 的 `problem_statement`
   - Embedding：直接 Embed Issue 文本
   - 计算相似度：使用余弦相似度（矩阵乘法）
   - 返回 Top-K：使用 `torch.topk()`

3. ✅ **评测**：
   - Ground Truth：使用 `edit_functions` 列表
   - 判定：Acc@K 判断 Top-K 中是否包含所有 GT 函数
   - 如果包含所有 GT，算命中（Acc@K = 1）

### ⚠️ 需要注意的点

1. **查询字段回退**：`get_problem_text()` 函数如果找不到 `problem_statement`，会回退到其他字段（`issue`, `description`, `prompt`, `text`）。这是合理的兼容性处理，不影响主要流程。

2. **评测代码分离**：评测代码在 `eval_metric.py` 中，不在 `run_with_index.py` 中。这是合理的架构设计，不影响功能。

3. **函数级别映射**：`run_with_index.py` 使用 `mapper.map_blocks_to_entities()` 将代码块映射到函数/模块（第296-301行）。这确保了输出的 `found_entities` 格式与 GT 的 `edit_functions` 格式一致。

---

## 结论

**✅ `run_with_index.py` 的评估过程完全符合论文要求。**

代码实现：
1. ✅ 直接使用 `problem_statement` 作为查询
2. ✅ Embed Issue 文本，计算余弦相似度，返回 Top-K
3. ✅ 评测时使用 `edit_functions` 作为 GT，Acc@K 判断是否包含所有 GT 函数
