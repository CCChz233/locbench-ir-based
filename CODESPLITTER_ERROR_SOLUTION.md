# CodeSplitter 错误解决方案

## 问题诊断

从日志可以看到，两个仓库无法使用 `llamaindex_code` 策略构建索引：

```
[DEBUG] pylint-dev_pylint: CodeSplitter Exception: ValueError: Could not parse code with language python.
[DEBUG] ckan_ckan: CodeSplitter Exception: ValueError: Could not parse code with language python.
```

## 根本原因

`CodeSplitter` 使用的 tree-sitter Python 解析器无法解析这两个仓库中的某些代码文件。

可能的原因：
1. 某些文件有语法错误或格式问题
2. tree-sitter 解析器对某些代码结构无法处理
3. 文件编码或格式问题

## 解决方案

已在代码中添加**降级处理（Fallback）机制**：

当 `CodeSplitter` 失败时，自动使用 `fixed` 分块策略作为备用方案。

### fixed 策略的特点

- **不依赖 AST 解析**：使用固定行数的非重叠块（默认40行）
- **更稳健**：可以处理任何格式的代码文件
- **兼容性好**：不依赖 tree-sitter 等外部解析器

### 实现逻辑

```python
except Exception as e:
    logger.error(f"LlamaIndex CodeSplitter failed for {repo_path}: {e}")
    logger.warning(f"Falling back to fixed chunking strategy for {repo_path}")
    # 使用 fixed 分块策略作为备用
    blocks = collect_blocks(repo_path, "fixed", block_size=40, window_size=50, slice_size=10)
    return blocks
```

## 结果

现在重新运行构建脚本，这两个仓库应该能使用 `fixed` 策略成功构建索引。

虽然分块方式不同（从 AST-based 降级到 fixed-line），但至少可以生成索引，确保所有165个仓库都能被索引。

## 后续优化建议

如果需要保持一致的 `llamaindex_code` 策略，可以考虑：

1. **检查并修复问题文件**：找出导致解析失败的具体文件
2. **使用更宽松的解析设置**：调整 CodeSplitter 的参数
3. **预处理文件**：在解析前清理或修复有问题的文件
