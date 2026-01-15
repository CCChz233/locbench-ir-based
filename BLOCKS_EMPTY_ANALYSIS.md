# Blocks为空的问题分析

## 日志输出

从用户提供的日志可以看到：
```
[Process 0] ⚠️  Warning: No blocks generated for pylint-dev_pylint, skipping.
[Process 0] ⚠️  Warning: No blocks generated for ckan_ckan, skipping.
```

这说明 `collect_llamaindex_code_blocks` 函数返回了空的blocks列表。

## 代码执行流程

`collect_llamaindex_code_blocks` 函数的执行流程：

1. **SimpleDirectoryReader 加载文档**（第1555行）
   - 使用排除规则：`**/test/**`, `**/tests/**`, `**/test_*.py`, `**/*_test.py`
   - 手动测试显示可以加载479个文档（对于ckan_ckan）
   - ✅ 已添加日志：`Loaded {len(docs)} documents from {repo_path}`

2. **CodeSplitter 生成节点**（第1569行）
   - 需要 `tree_sitter_language_pack` 依赖
   - ✅ 已添加日志：`Generated {len(nodes)} nodes from {len(docs)} documents for {repo_path}`

3. **节点转换为Blocks**（第1593-1622行）
   - 遍历所有nodes
   - 过滤条件：
     - `if not file_path: continue`（第1595行）
     - `if not content_text: continue`（第1599行）
   - ✅ 已添加日志：`No blocks created from {len(nodes)} nodes for {repo_path}`

## 可能的原因

### 1. CodeSplitter 返回空nodes
- CodeSplitter处理失败但没有抛出异常
- 所有文档都无法被解析成节点

### 2. 节点转换时全部被过滤
- 所有节点的 `file_path` 为空
- 或所有节点的 `content_text` 为空

## 下一步

重新运行构建脚本，查看详细的日志输出：

```bash
cd /workspace/locbench/IR-based
python method/index/batch_build_index.py \
    --repo_path /workspace/locbench/repos/locbench_repos \
    --strategy llamaindex_code \
    --index_dir new_index_data/llamacode_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 1 \
    --gpu_ids 1 \
    --batch_size 32 2>&1 | tee build_log.txt
```

然后在日志中搜索这两个仓库的名称，查看：
- `Loaded X documents from ...` - 加载了多少文档
- `Generated X nodes from ...` - 生成了多少节点
- `No blocks created from X nodes for ...` - 是否所有节点都被过滤
- 任何错误或警告信息

这些日志信息将帮助我们定位具体是哪个步骤出了问题。
