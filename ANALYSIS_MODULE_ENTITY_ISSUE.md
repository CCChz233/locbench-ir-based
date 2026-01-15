# 模块和函数级别指标为0%的原因分析

## 问题现象

在 `ir_function_CodeRankEmbed` 策略的评估结果中：
- **模块级别**：Recall@5 和 Recall@10 均为 0%
- **函数级别**：Recall@5 和 Recall@10 均为 0%

## 根本原因

通过分析代码和测试，发现问题的根本原因是：

### 1. 输出文件中 `found_modules` 和 `found_entities` 全部为空

检查输出文件 `new_outputs/ir_function_CodeRankEmbed/loc_outputs.jsonl`：
- 所有560个实例的 `found_modules` 都是空数组 `[]`
- 所有560个实例的 `found_entities` 都是空数组 `[]`

### 2. Mapper 功能正常

通过独立测试 `ASTBasedMapper`，确认：
- Mapper 能够正确解析AST并提取模块和实体
- 给定正确的代码块信息，mapper 能够返回模块和实体列表

测试结果示例：
```
找到的模块数: 7
找到的实体数: 10
模块示例: ['uxarray/formatting_html.py:_grid_header', ...]
实体示例: ['uxarray/formatting_html.py:_grid_header', ...]
```

### 3. 可能的原因

#### 原因A：代码块行号格式问题

在 `run_with_index.py` 中，代码块的行号是 **0-indexed**（从0开始），但 AST 解析器期望的是 **1-indexed**（从1开始）。

查看代码：
```python
# run_with_index.py 第317行
found_modules, found_entities = mapper.map_blocks_to_entities(
    blocks=top_blocks,  # top_blocks 中的 start_line/end_line 是 0-indexed
    instance_id=instance_id,
    top_k_modules=args.top_k_modules,
    top_k_entities=args.top_k_entities,
)
```

而在 `ASTBasedMapper.map_blocks_to_entities` 中：
```python
# mapper.py 第193-194行
ast_start = block_start + 1  # 转换为 1-indexed
ast_end = block_end + 1
```

**但是**，如果 `block_start` 或 `block_end` 已经是负数或无效值，转换后仍然无效。

#### 原因B：文件路径不匹配

`clean_file_path` 函数可能没有正确处理所有路径格式，导致mapper无法找到对应的文件。

#### 原因C：异常被静默处理

在 `run_with_index.py` 中，如果 mapper 抛出异常，可能被静默处理，导致返回空列表。

## 验证步骤

### 步骤1：检查实际运行时的代码块数据

```python
# 在 run_with_index.py 中添加调试输出
print(f"Top blocks before mapping: {top_blocks[:3]}")
found_modules, found_entities = mapper.map_blocks_to_entities(...)
print(f"After mapping: modules={len(found_modules)}, entities={len(found_entities)}")
```

### 步骤2：检查是否有异常

```python
try:
    found_modules, found_entities = mapper.map_blocks_to_entities(...)
except Exception as e:
    print(f"Mapper error: {e}")
    import traceback
    traceback.print_exc()
```

### 步骤3：验证文件路径

检查 `clean_file_path` 是否正确处理了所有路径格式。

## 解决方案

### 方案1：添加调试日志（推荐）

在 `run_with_index.py` 的 mapper 调用处添加详细的日志：

```python
# 在 run_with_index.py 第317行附近
logging.debug(f"Mapping blocks for {instance_id}: {len(top_blocks)} blocks")
for i, block in enumerate(top_blocks[:3]):
    logging.debug(f"  Block {i}: file_path={block.get('file_path')}, "
                  f"start_line={block.get('start_line')}, end_line={block.get('end_line')}")

try:
    found_modules, found_entities = mapper.map_blocks_to_entities(
        blocks=top_blocks,
        instance_id=instance_id,
        top_k_modules=args.top_k_modules,
        top_k_entities=args.top_k_entities,
    )
    logging.debug(f"Mapper result: {len(found_modules)} modules, {len(found_entities)} entities")
except Exception as e:
    logging.error(f"Mapper failed for {instance_id}: {e}", exc_info=True)
    found_modules, found_entities = [], []
```

### 方案2：修复行号转换

确保行号正确转换：

```python
# 在传递给mapper之前，确保行号有效
for block in top_blocks:
    if block.get('start_line', -1) < 0 or block.get('end_line', -1) < 0:
        logging.warning(f"Invalid line numbers in block: {block}")
        # 跳过或修复
```

### 方案3：验证文件路径

确保 `clean_file_path` 正确处理所有情况：

```python
# 在 clean_file_path 中添加日志
def clean_file_path(file_path: str, repo_name: str) -> str:
    original = file_path
    # ... 清理逻辑 ...
    if original != result:
        logging.debug(f"Cleaned path: {original} -> {result}")
    return result
```

## 下一步行动

1. **立即行动**：在 `run_with_index.py` 中添加调试日志，重新运行一个实例，查看实际传递给mapper的数据
2. **检查异常**：确认是否有异常被静默处理
3. **验证路径**：确认文件路径是否正确清理
4. **修复问题**：根据调试结果修复具体问题

## 预期结果

修复后，应该能看到：
- `found_modules` 包含模块ID列表（如 `['file.py:ClassName', ...]`）
- `found_entities` 包含实体ID列表（如 `['file.py:function_name', ...]`）
- 模块和函数级别的Recall指标 > 0%
