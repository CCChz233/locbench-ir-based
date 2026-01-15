# 完整分块策略分析

根据 `method/index/batch_build_index.py` 文件分析，IR-based 框架支持 **13种分块策略**。

## 策略分类

### 1. 基础策略（4种）- 行级切块

这4种策略在 `collect_blocks()` 函数中实现，基于行号进行切块。

#### 1.1 `fixed` - 固定行切块
- **函数**: `blocks_fixed_lines()`
- **特点**: 按固定行数切分，不重叠
- **参数**: `--block_size 15`（默认15行）
- **实现位置**: 第117-127行

#### 1.2 `sliding` - 滑动窗口切块
- **函数**: `blocks_sliding()`
- **特点**: 使用滑动窗口，可以重叠
- **参数**: `--window_size 20 --slice_size 2`（默认窗口20行，步长10行）
- **实现位置**: 第130-141行

#### 1.3 `rl_fixed` - RLCoder 固定块
- **函数**: `blocks_rl_fixed()`
- **特点**: 按非空行数切分，每块12个非空行
- **参数**: 无（固定12非空行）
- **实现位置**: 第144-152行

#### 1.4 `rl_mini` - RLCoder Mini块
- **函数**: `blocks_rl_mini()`
- **特点**: 按空行分段，拼接成≤15行的块
- **参数**: 无（算法内部处理）
- **实现位置**: 第155-193行

---

### 2. 函数级策略（2种）- AST解析

这2种策略使用AST解析提取函数和方法。

#### 2.1 `function_level` - 函数级切块（带fallback）
- **函数**: `collect_function_blocks()` → `blocks_function_level_with_fallback()`
- **特点**: 
  - 使用AST提取所有函数和方法
  - 未覆盖的代码（imports、全局变量等）使用fixed策略补充
  - 确保完整覆盖所有代码
- **参数**: `--block_size 15`（fallback使用）
- **实现位置**: 第1908-2024行（collect_function_blocks）
- **依赖函数**: 
  - `blocks_by_function()` (第697-903行)
  - `blocks_function_level_with_fallback()` (第600-694行)

#### 2.2 `ir_function` - IR-based 论文方法（函数级）
- **函数**: `collect_ir_function_blocks()` → `blocks_ir_function()`
- **特点**:
  - 按照论文 IR-based 方法实现
  - 每个函数单独作为一个嵌入单元
  - 模块级代码（imports、全局变量）冗余复制到该文件每个函数
  - 类属性冗余复制到该类每个方法
  - 只索引Python文件中的函数
- **参数**: 无
- **实现位置**: 第470-560行（collect_ir_function_blocks）
- **依赖函数**: 
  - `blocks_ir_function()` (第322-467行)
  - `_build_ir_function_representation()` (第277-319行)
  - `_extract_module_level_code()` (第204-241行)
  - `_extract_class_attributes()` (第244-274行)

---

### 3. Epic策略（1种）- 语义切块

#### 3.1 `epic` - EpicSplitter（与BM25一致）
- **函数**: `collect_epic_blocks()`
- **特点**: 
  - 使用与BM25相同的EpicSplitter
  - 基于代码语义的智能切块
  - 如果失败会回退到rl_mini策略
- **参数**: 通过 `EpicSplitterConfig` 配置（代码内部）
  - `min_chunk_size: 512`
  - `chunk_size: 1024`
  - `max_chunk_size: 2048`
  - `hard_token_limit: 4096`
  - `max_chunks: 512`
- **实现位置**: 第972-1069行
- **依赖**: `repo_index.index.epic_split.EpicSplitter`

---

### 4. LangChain策略（3种）- 文本分割器

这3种策略使用LangChain的文本分割器。

#### 4.1 `langchain_fixed` - CharacterTextSplitter
- **函数**: `collect_langchain_fixed_blocks()`
- **特点**: 按固定字符数分割，使用`\n\n`作为分隔符
- **参数**: 
  - `--langchain_chunk_size 1000`（默认1000字符）
  - `--langchain_chunk_overlap 200`（默认200字符重叠）
- **实现位置**: 第1279-1381行
- **依赖**: `langchain_text_splitters.CharacterTextSplitter`

#### 4.2 `langchain_recursive` - RecursiveCharacterTextSplitter
- **函数**: `collect_langchain_recursive_blocks()`
- **特点**: 递归字符分割，尝试保持代码结构
- **参数**: 
  - `--langchain_chunk_size 1000`（默认1000字符）
  - `--langchain_chunk_overlap 200`（默认200字符重叠）
- **实现位置**: 第1175-1276行
- **依赖**: `langchain_text_splitters.RecursiveCharacterTextSplitter`

#### 4.3 `langchain_token` - TokenTextSplitter
- **函数**: `collect_langchain_token_blocks()`
- **特点**: 按token数分割
- **参数**: 
  - `--langchain_chunk_size 1000`（默认1000 tokens）
  - `--langchain_chunk_overlap 200`（默认200 tokens重叠）
- **实现位置**: 第1384-1486行
- **依赖**: `langchain_text_splitters.TokenTextSplitter`

---

### 5. LlamaIndex策略（4种）- 代码/语义分割器

这4种策略使用LlamaIndex的分割器。

#### 5.1 `llamaindex_code` - CodeSplitter
- **函数**: `collect_llamaindex_code_blocks()`
- **特点**: 代码专用分割器，基于AST
- **参数**: 
  - `--llamaindex_language python`（默认python）
  - `--llamaindex_chunk_lines 40`（默认40行）
  - `--llamaindex_chunk_lines_overlap 15`（默认15行重叠）
  - `--llamaindex_max_chars 1500`（默认1500字符）
- **实现位置**: 第1489-1593行
- **依赖**: `llama_index.core.node_parser.CodeSplitter`

#### 5.2 `llamaindex_sentence` - SentenceSplitter
- **函数**: `collect_llamaindex_sentence_blocks()`
- **特点**: 按句子分割
- **参数**: 
  - `--llamaindex_chunk_size 1024`（默认1024字符）
  - `--llamaindex_chunk_overlap 200`（默认200字符重叠）
- **实现位置**: 第1596-1694行
- **依赖**: `llama_index.core.node_parser.SentenceSplitter`

#### 5.3 `llamaindex_token` - TokenTextSplitter（LlamaIndex版本）
- **函数**: `collect_llamaindex_token_blocks()`
- **特点**: 按token数分割（LlamaIndex实现）
- **参数**: 
  - `--llamaindex_chunk_size 1024`（默认1024 tokens）
  - `--llamaindex_chunk_overlap 20`（默认20 tokens重叠）
  - `--llamaindex_separator " "`（默认空格分隔符）
- **实现位置**: 第1697-1798行
- **依赖**: `llama_index.core.node_parser.TokenTextSplitter`

#### 5.4 `llamaindex_semantic` - SemanticSplitterNodeParser
- **函数**: `collect_llamaindex_semantic_blocks()`
- **特点**: 基于语义相似性分割
- **参数**: 
  - `--llamaindex_buffer_size 3`（默认3）
  - `--llamaindex_embed_model sentence-transformers/all-MiniLM-L6-v2`（默认轻量级模型）
- **实现位置**: 第1801-1905行
- **依赖**: 
  - `llama_index.core.node_parser.SemanticSplitterNodeParser`
  - `llama_index.embeddings.huggingface.HuggingFaceEmbedding`

---

## 完整策略列表（13种）

| 序号 | 策略名称 | 类别 | 特点 | 主要参数 |
|------|----------|------|------|----------|
| 1 | `fixed` | 基础策略 | 固定行数，不重叠 | `--block_size` |
| 2 | `sliding` | 基础策略 | 滑动窗口，可重叠 | `--window_size --slice_size` |
| 3 | `rl_fixed` | 基础策略 | 12非空行/块 | 无 |
| 4 | `rl_mini` | 基础策略 | 按空行分段，≤15行/块 | 无 |
| 5 | `function_level` | 函数级 | AST提取函数+fallback补充 | `--block_size` |
| 6 | `ir_function` | 函数级 | 论文IR方法，上下文冗余复制 | 无 |
| 7 | `epic` | 语义切块 | EpicSplitter（与BM25一致） | 代码内部配置 |
| 8 | `langchain_fixed` | LangChain | 固定字符数分割 | `--langchain_chunk_size --langchain_chunk_overlap` |
| 9 | `langchain_recursive` | LangChain | 递归字符分割 | `--langchain_chunk_size --langchain_chunk_overlap` |
| 10 | `langchain_token` | LangChain | Token数分割 | `--langchain_chunk_size --langchain_chunk_overlap` |
| 11 | `llamaindex_code` | LlamaIndex | 代码专用，基于AST | `--llamaindex_*` 多个参数 |
| 12 | `llamaindex_sentence` | LlamaIndex | 按句子分割 | `--llamaindex_chunk_size --llamaindex_chunk_overlap` |
| 13 | `llamaindex_token` | LlamaIndex | Token数分割 | `--llamaindex_chunk_size --llamaindex_chunk_overlap --llamaindex_separator` |
| 14 | `llamaindex_semantic` | LlamaIndex | 基于语义相似性 | `--llamaindex_buffer_size --llamaindex_embed_model` |

**注意**: 实际上有14种策略，但在argparse的choices中列出了13种（因为有些策略有多个变体）。

---

## 策略选择流程

在 `run()` 函数中（第2282-2343行），策略选择逻辑如下：

```python
if args.strategy == "epic":
    blocks = collect_epic_blocks(...)
elif args.strategy == "function_level":
    blocks, function_metadata = collect_function_blocks(...)
elif args.strategy == "ir_function":
    blocks, function_metadata = collect_ir_function_blocks(...)
elif args.strategy == "langchain_fixed":
    blocks = collect_langchain_fixed_blocks(...)
elif args.strategy == "langchain_recursive":
    blocks = collect_langchain_recursive_blocks(...)
elif args.strategy == "langchain_token":
    blocks = collect_langchain_token_blocks(...)
elif args.strategy == "llamaindex_code":
    blocks = collect_llamaindex_code_blocks(...)
elif args.strategy == "llamaindex_sentence":
    blocks = collect_llamaindex_sentence_blocks(...)
elif args.strategy == "llamaindex_token":
    blocks = collect_llamaindex_token_blocks(...)
elif args.strategy == "llamaindex_semantic":
    blocks = collect_llamaindex_semantic_blocks(...)
else:
    # 基础策略：fixed, sliding, rl_fixed, rl_mini
    blocks = collect_blocks(...)
```

---

## 使用示例

### 基础策略

```bash
# fixed
python method/index/batch_build_index.py --strategy fixed --block_size 15

# sliding
python method/index/batch_build_index.py --strategy sliding --window_size 20 --slice_size 2

# rl_fixed
python method/index/batch_build_index.py --strategy rl_fixed

# rl_mini
python method/index/batch_build_index.py --strategy rl_mini
```

### 函数级策略

```bash
# function_level
python method/index/batch_build_index.py --strategy function_level --block_size 15

# ir_function（论文方法）
python method/index/batch_build_index.py --strategy ir_function
```

### Epic策略

```bash
python method/index/batch_build_index.py --strategy epic
```

### LangChain策略

```bash
# langchain_fixed
python method/index/batch_build_index.py --strategy langchain_fixed \
    --langchain_chunk_size 1000 --langchain_chunk_overlap 200

# langchain_recursive
python method/index/batch_build_index.py --strategy langchain_recursive \
    --langchain_chunk_size 1000 --langchain_chunk_overlap 200

# langchain_token
python method/index/batch_build_index.py --strategy langchain_token \
    --langchain_chunk_size 1000 --langchain_chunk_overlap 200
```

### LlamaIndex策略

```bash
# llamaindex_code
python method/index/batch_build_index.py --strategy llamaindex_code \
    --llamaindex_language python --llamaindex_chunk_lines 40 \
    --llamaindex_chunk_lines_overlap 15 --llamaindex_max_chars 1500

# llamaindex_sentence
python method/index/batch_build_index.py --strategy llamaindex_sentence \
    --llamaindex_chunk_size 1024 --llamaindex_chunk_overlap 200

# llamaindex_token
python method/index/batch_build_index.py --strategy llamaindex_token \
    --llamaindex_chunk_size 1024 --llamaindex_chunk_overlap 20

# llamaindex_semantic
python method/index/batch_build_index.py --strategy llamaindex_semantic \
    --llamaindex_buffer_size 3 \
    --llamaindex_embed_model sentence-transformers/all-MiniLM-L6-v2
```

---

## 策略对比总结

| 类别 | 策略数量 | 特点 | 适用场景 |
|------|----------|------|----------|
| **基础策略** | 4种 | 简单高效，基于行号 | 快速基线、对比实验 |
| **函数级策略** | 2种 | AST解析，语义单元 | 函数级检索、代码结构理解 |
| **Epic策略** | 1种 | 语义切块，与BM25一致 | 与BM25对比实验 |
| **LangChain策略** | 3种 | 通用文本分割 | 文本处理、多语言支持 |
| **LlamaIndex策略** | 4种 | 代码/语义专用 | 代码理解、语义检索 |
| **总计** | **14种** | - | - |

---

## 注意事项

1. **函数级策略返回metadata**: `function_level` 和 `ir_function` 策略返回 `(blocks, function_metadata)`，其他策略只返回 `blocks`

2. **Epic策略有fallback**: 如果EpicSplitter失败，会自动回退到 `rl_mini` 策略

3. **LangChain和LlamaIndex策略**: 需要相应的依赖包（`langchain-text-splitters`, `llama-index-core`等）

4. **索引输出目录**: 所有策略的索引都保存在 `index_data/{dataset_name}/dense_index_{strategy}/` 目录下

5. **span_ids支持**: 如果提供了 `--graph_index_dir`，所有策略都可以在metadata中添加 `span_ids`（用于Graph映射器）
