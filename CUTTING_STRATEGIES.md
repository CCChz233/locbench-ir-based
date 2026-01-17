# 代码切块策略总结

本文档总结了 `batch_build_index.py` 中支持的所有代码切块策略。

## 目录

1. [基础策略](#基础策略)
2. [函数级策略](#函数级策略)
3. [高级策略](#高级策略)
4. [LangChain策略](#langchain策略)
5. [LlamaIndex策略](#llamaindex策略)

---

## 基础策略

### 1. Fixed（固定行数切块）

**策略名**: `fixed`

**函数**: `blocks_fixed_lines(text, rel, block_size)`

**原理**:
- 按照固定的行数（默认15行）将代码分割成块
- 不重叠，连续切块
- 跳过全空块

**实现**:
```python
def blocks_fixed_lines(text: str, rel: str, block_size: int) -> List[Block]:
    lines = text.splitlines()
    blocks: List[Block] = []
    start = 0
    while start < len(lines):
        end = min(start + block_size, len(lines))
        chunk = lines[start:end]
        if any(l.strip() for l in chunk):
            blocks.append(Block(rel, start, end - 1, "\n".join(chunk), "fixed"))
        start = end
    return blocks
```

**特点**:
- ✅ 简单高效
- ✅ 无重叠，覆盖完整
- ❌ 可能破坏函数/类边界
- ❌ 不考虑代码语义

**参数**:
- `--block_size`: 块大小（行数），默认15

---

### 2. Sliding（滑动窗口切块）

**策略名**: `sliding`

**函数**: `blocks_sliding(text, rel, window_size, slice_size)`

**原理**:
- 使用滑动窗口（默认20行）提取代码块
- 每隔 `window_size // slice_size` 行提取一个窗口
- 窗口重叠：窗口大小的50%

**实现**:
```python
def blocks_sliding(text: str, rel: str, window_size: int, slice_size: int) -> List[Block]:
    lines = text.splitlines()
    delta = window_size // 2  # 重叠50%
    step = max(1, window_size // slice_size) if slice_size > 0 else window_size
    blocks: List[Block] = []
    for line_no in range(0, len(lines), step):
        start = max(0, line_no - delta)
        end = min(len(lines), line_no + window_size - delta)
        chunk = lines[start:end]
        if any(l.strip() for l in chunk):
            blocks.append(Block(rel, start, end - 1, "\n".join(chunk), "sliding"))
    return blocks
```

**特点**:
- ✅ 有重叠，更好的上下文
- ✅ 捕捉更多语义边界
- ❌ 产生更多块，索引变大
- ❌ 计算成本更高

**参数**:
- `--window_size`: 窗口大小（行数），默认20
- `--slice_size`: 步长因子，默认2

---

### 3. RL-Fixed（RL论文的固定策略）

**策略名**: `rl_fixed`

**函数**: `blocks_rl_fixed(text, rel, max_lines)`

**原理**:
- 模仿 RLRetriever 论文的切块方式
- 每12个非空行组成一个块
- 限制最多处理5000行

**实现**:
```python
def blocks_rl_fixed(text: str, rel: str, max_lines: int = 5000) -> List[Block]:
    lines = text.splitlines()
    non_empty = [(idx, line) for idx, line in enumerate(lines) if line.strip()]
    blocks: List[Block] = []
    for i in range(0, min(len(non_empty), max_lines), 12):
        chunk = non_empty[i:i + 12]
        if not chunk:
            break
        start = chunk[0][0]
        end = chunk[-1][0]
        content = "\n".join(line for _, line in chunk)
        blocks.append(Block(rel, start, end, content, "rl_fixed"))
    return blocks
```

**特点**:
- ✅ 基于论文实现
- ✅ 跳过空行
- ❌ 跨度大，可能包含不相关代码
- ❌ 不保留文件结构

**参数**:
- 无（内部固定：每块12个非空行）

---

### 4. RL-Mini（RL论文的Mini策略）

**策略名**: `rl_mini`

**函数**: `blocks_rl_mini(text, rel, max_lines)`

**原理**:
- RL论文的另一个切块方式
- 优先按空行分组
- 每组最多15行，超过则拆分

**实现**:
```python
def blocks_rl_mini(text: str, rel: str, max_lines: int = 5000) -> List[Block]:
    mini_blocks = []
    cur = []
    for idx, line in enumerate(text.splitlines()):
        if line.strip():
            cur.append((idx, line))
        else:
            if cur:
                mini_blocks.append(cur)
                cur = []
    if cur:
        mini_blocks.append(cur)

    temp = []
    for mb in mini_blocks:
        if len(mb) > 15:
            for idx in range(0, len(mb), 15):
                temp.append(mb[idx: idx + 15])
        else:
            temp.append(mb)
    mini_blocks = temp

    blocks: List[Block] = []
    current = []
    total = 0
    for block in mini_blocks:
        if total >= max_lines:
            break
        if len(current) + len(block) <= 15:
            current.extend(block)
            total += len(block)
        else:
            if current:
                start = current[0][0]
                end = current[-1][0]
                content = "\n".join(line for _, line in current)
                blocks.append(Block(rel, start, end, content, "rl_mini"))
            current = list(block)
            total += len(block)
    if current:
        start = current[0][0]
        end = current[-1][0]
        content = "\n".join(line for _, line in current)
        blocks.append(Block(rel, start, end, content, "rl_mini"))
    return blocks
```

**特点**:
- ✅ 利用空行作为语义边界
- ✅ 基于论文实现
- ❌ 可能遗漏空行内的代码
- ❌ 复杂度高

**参数**:
- 无（内部固定）

---

## 函数级策略

### 5. Function-Level（函数级切块 + Fallback）

**策略名**: `function_level`

**函数**: `blocks_function_level_with_fallback(text, rel, block_size)`

**原理**:
- 使用AST解析提取所有函数和类方法
- 未被函数覆盖的代码（imports、全局变量、类属性）使用fixed切块补充
- 每个函数添加格式：`# qualified_name\n\n{code}`

**实现**:
```python
def blocks_function_level_with_fallback(text: str, rel: str, block_size: int = 15):
    # 1. 先提取所有函数级块
    function_blocks, function_metadata = blocks_by_function(text, rel)
    
    # 2. 记录被函数覆盖的行号
    lines = text.splitlines()
    covered_lines = set()
    for block in function_blocks:
        for line_no in range(block.start, block.end + 1):
            covered_lines.add(line_no)
    
    # 3. 找出未被覆盖的连续行区间
    # 4. 对未覆盖的区间使用 fixed 策略切分
    
    # 5. 合并函数块和补充块
    all_blocks = function_blocks + fallback_blocks
    return all_blocks
```

**特点**:
- ✅ 保留函数语义边界
- ✅ 完整覆盖所有代码
- ✅ 函数有qualified_name标识
- ❌ 只支持Python
- ❌ 非Python文件回退到fixed

**参数**:
- `--block_size`: 对未覆盖代码使用的fixed切块大小，默认15行

**元数据**:
```json
{
  "block_id": 0,
  "file_path": "src/utils.py",
  "start_line": 10,
  "end_line": 25,
  "block_type": "function_level",
  "strategy": "function_level",
  "qualified_name": "src/utils.py::MathUtils::calculate_sum",
  "class_name": "MathUtils",
  "function_name": "calculate_sum",
  "function_level_version": "1.0"
}
```

---

### 6. IR-Function（IR论文的函数级切块）

**策略名**: `ir_function`

**函数**: `blocks_ir_function(text, rel) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]`

**原理**:
- 按照论文 IR-based 方法（Appendix C.1.1）实现
- 每个函数单独作为一个嵌入单元
- **模块级代码（imports、全局变量）冗余复制到该文件每个函数**
- **类属性冗余复制到该类每个方法**
- 格式：`{file_path} {class_name} {module_code} {class_attrs} {function_code}`

**实现**:
```python
def blocks_ir_function(text: str, rel: str) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]:
    # 1. 提取模块级代码（imports、全局变量、常量定义）
    module_level_code = _extract_module_level_code(tree, lines)
    
    # 2. 遍历AST提取函数
    # 对于每个函数：
    #   - 提取函数代码（包括装饰器）
    #   - 如果是类方法，提取类属性
    #   - 构建完整表示：file_path + class_name + module_code + class_attrs + function_code
```

**与 function_level 的区别**:

| 特性 | function_level | ir_function |
|-------|---------------|-------------|
| 格式 | `# qualified_name\n\n{code}` | `{file_path} {class} {module} {attrs} {code}` |
| 上下文 | 只函数本身 | 冗余的文件、类、模块级代码 |
| 论文基础 | 无 | IR-based 论文复现 |
| 索引结构 | Flat | Flat（无单独的全局变量索引） |

**特点**:
- ✅ 论文复现（IR-based Appendix C.1.1）
- ✅ 上下文冗余增强表示
- ✅ 只索引函数（非函数代码被忽略）
- ❌ 只支持Python
- ❌ 冗余可能增加噪声

**参数**:
- `--ir_function_context_tokens`: 上下文token上限，默认256

**元数据**:
```json
{
  "block_id": 0,
  "file_path": "src/utils.py",
  "start_line": 10,
  "end_line": 25,
  "block_type": "ir_function",
  "strategy": "ir_function",
  "qualified_name": "src/utils.py::MathUtils::calculate_sum",
  "class_name": "MathUtils",
  "function_name": "calculate_sum",
  "ir_function_version": "1.0"
}
```

---

## 高级策略

### 7. Epic（BM25一致的切块）

**策略名**: `epic`

**函数**: `collect_epic_blocks(repo_path, config)`

**原理**:
- 使用 EpicSplitter（与BM25保持一致）
- 智能切块：min_chunk_size ~ chunk_size ~ max_chunk_size
- 基于语义和token数动态调整

**参数**（EpicSplitterConfig）:
```python
@dataclass
class EpicSplitterConfig:
    min_chunk_size: int = 512      # 最小块大小
    chunk_size: int = 1024          # 目标块大小
    max_chunk_size: int = 2048      # 最大块大小
    hard_token_limit: int = 4096    # 硬性token限制
    max_chunks: int = 512          # 每个文件最大块数
```

**特点**:
- ✅ 与BM25索引完全一致
- ✅ 自适应块大小
- ✅ 上下文增强
- ❌ 复杂度高
- ❌ 需要额外依赖

**参数**:
- 无（使用EpicSplitterConfig默认值）

**元数据**:
```json
{
  "block_id": 0,
  "file_path": "src/utils.py",
  "start_line": 0,
  "end_line": 25,
  "block_type": "epic",
  "strategy": "epic",
  "epic_splitter_version": "1.0",
  "chunking_config": {
    "min_chunk_size": 512,
    "chunk_size": 1024,
    "max_chunk_size": 2048,
    "hard_token_limit": 4096,
    "max_chunks": 512
  },
  "context_enhanced": true,
  "chunk_tokens": 150,
  "created_at": "2026-01-17T00:00:00"
}
```

---

## LangChain策略

### 8. LangChain-Fixed（固定字符切块）

**策略名**: `langchain_fixed`

**函数**: `collect_langchain_fixed_blocks(repo_path, chunk_size, chunk_overlap)`

**原理**:
- 使用 LangChain 的 CharacterTextSplitter
- 按 `\n\n` 分隔符切分
- 每块最大字符数（默认1000），重叠字符数（默认200）

**特点**:
- ✅ 基于段落结构
- ✅ 简单可控
- ❌ 可能破坏函数边界
- ❌ 基于字符而非语义

**参数**:
- `--langchain_chunk_size`: 目标块大小（字符数），默认1000
- `--langchain_chunk_overlap`: 块之间的重叠字符数，默认200

---

### 9. LangChain-Recursive（递归字符切块）

**策略名**: `langchain_recursive`

**函数**: `collect_langchain_recursive_blocks(repo_path, chunk_size, chunk_overlap)`

**原理**:
- 使用 LangChain 的 RecursiveCharacterTextSplitter
- 递归尝试多种分隔符：`\n\n` → `\n` → `' '`
- 优先保留语义边界

**特点**:
- ✅ 智能分隔符选择
- ✅ 保留更多语义结构
- ❌ 复杂度更高
- ❌ 可能产生过多小块

**参数**:
- `--langchain_chunk_size`: 目标块大小（字符数），默认1000
- `--langchain_chunk_overlap`: 块之间的重叠字符数，默认200

---

### 10. LangChain-Token（Token切块）

**策略名**: `langchain_token`

**函数**: `collect_langchain_token_blocks(repo_path, chunk_size, chunk_overlap)`

**原理**:
- 使用 LangChain 的 TokenTextSplitter
- 按token数（而非字符数）切分
- 基于tokenizer的token数量

**特点**:
- ✅ 更精确控制token数量
- ✅ 适合模型输入限制
- ❌ 需要tokenizer
- ❌ 计算开销大

**参数**:
- `--langchain_chunk_size`: 目标块大小（token数），默认1000
- `--langchain_chunk_overlap`: 块之间的重叠token数，默认200

---

## LlamaIndex策略

### 11. LlamaIndex-Code（代码专用切块）

**策略名**: `llamaindex_code`

**函数**: `collect_llamaindex_code_blocks(repo_path, language, chunk_lines, chunk_lines_overlap, max_chars)`

**原理**:
- 使用 LlamaIndex 的 CodeSplitter
- **基于AST**，保留代码结构
- 按代码块（函数、类、语句）切分
- 每块最多N行（默认40），重叠M行（默认15），最多P字符（默认1500）

**特点**:
- ✅ **基于AST，保留代码结构**
- ✅ 不破坏函数/类边界
- ✅ 适合代码检索
- ❌ 只支持解析的编程语言
- ❌ 复杂度高

**参数**:
- `--llamaindex_language`: 编程语言，默认"python"
- `--llamaindex_chunk_lines`: 每个块的代码行数，默认40
- `--llamaindex_chunk_lines_overlap`: 块之间的重叠行数，默认15
- `--llamaindex_max_chars`: 每个块的最大字符数，默认1500

**元数据**:
```json
{
  "block_id": 0,
  "file_path": "src/utils.py",
  "start_line": 0,
  "end_line": 25,
  "block_type": "llamaindex_code",
  "strategy": "llamaindex_code"
}
```

---

### 12. LlamaIndex-Sentence（句子切块）

**策略名**: `llamaindex_sentence`

**函数**: `collect_llamaindex_sentence_blocks(repo_path, chunk_size, chunk_overlap)`

**原理**:
- 使用 LlamaIndex 的 SentenceSplitter
- 按句子切分（基于句号、分号等）
- 适合自然语言文档

**特点**:
- ✅ 保留句子结构
- ✅ 适合文档检索
- ❌ 不适合代码
- ❌ 可能在代码处错误切分

**参数**:
- `--llamaindex_chunk_size`: 目标块大小（字符数），默认1024
- `--llamaindex_chunk_overlap`: 块之间的重叠字符数，默认200

---

### 13. LlamaIndex-Token（Token切块）

**策略名**: `llamaindex_token`

**函数**: `collect_llamaindex_token_blocks(repo_path, chunk_size, chunk_overlap, separator)`

**原理**:
- 使用 LlamaIndex 的 TokenTextSplitter
- 按token数切分
- 自定义分隔符

**特点**:
- ✅ 精确控制token
- ✅ 适合模型输入
- ❌ 计算开销
- ❌ 可能破坏语义

**参数**:
- `--llamaindex_chunk_size`: 目标块大小（token数），默认1024
- `--llamaindex_chunk_overlap`: 块之间的重叠token数，默认20
- `--llamaindex_separator`: 分隔符，默认" "

---

### 14. LlamaIndex-Semantic（语义切块）

**策略名**: `llamaindex_semantic`

**函数**: `collect_llamaindex_semantic_blocks(repo_path, buffer_size, model_name)`

**原理**:
- 使用 LlamaIndex 的 SemanticSplitterNodeParser
- **基于嵌入相似性**切分
- 在语义变化大的地方分割
- 需要嵌入模型

**特点**:
- ✅ **语义感知**
- ✅ 保留语义边界
- ✅ 高质量切块
- ❌ 需要嵌入模型
- ❌ 计算成本高
- ❌ 依赖语义模型质量

**参数**:
- `--llamaindex_buffer_size`: 缓冲区大小，默认3
- `--llamaindex_embed_model`: HuggingFace嵌入模型名称，默认"sentence-transformers/all-MiniLM-L6-v2"

---

## AST使用分析

### 直接使用AST的策略（2种）

#### 1. function_level

**AST导入**: `import ast`（第38行）

**AST使用位置**: `blocks_by_function(text, rel)` 函数（第830-1031行）

**AST操作**:
```python
# 第848行：解析Python代码为AST
tree = ast.parse(text)
lines = text.splitlines()

# 第869-1001行：递归遍历AST节点
def visit_node(node: ast.AST):
    # 识别类定义
    if isinstance(node, ast.ClassDef):
        class_stack.append(node.name)
        for child in node.body:
            visit_node(child)
        class_stack.pop()
    
    # 识别函数定义
    elif isinstance(node, ast.FunctionDef):
        function_name = node.name
        function_stack.append(function_name)
        
        # 提取位置信息
        start_line = node.lineno - 1
        end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else start_line + 10
        
        # 提取装饰器
        if node.decorator_list:
            first_decorator = node.decorator_list[0]
            if hasattr(first_decorator, 'lineno'):
                start_line = first_decorator.lineno - 1
        
        # 递归处理嵌套函数
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_node(child)
```

**AST提取的信息**:
- ✅ 类定义（`ast.ClassDef`）
- ✅ 函数定义（`ast.FunctionDef`, `ast.AsyncFunctionDef`）
- ✅ 函数位置（`node.lineno`, `node.end_lineno`）
- ✅ 装饰器（`node.decorator_list`）
- ✅ 嵌套函数支持

---

#### 2. ir_function

**AST导入**: `import ast`（第38行）

**AST使用位置**: `blocks_ir_function(text, rel)` 函数（第413-578行）

**AST操作**:
```python
# 第435行：解析Python代码为AST
tree = ast.parse(text)
lines = text.splitlines()

# 第268-306行：提取模块级代码（imports、全局变量）
def _extract_module_level_code(tree: ast.AST, lines: List[str]) -> str:
    module_level_parts = []
    
    for node in tree.body:
        # 提取import语句
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            module_level_parts.append(code.strip())
        
        # 提取全局变量赋值
        elif isinstance(node, ast.Assign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            module_level_parts.append(code.strip())
        
        # 提取类型注解的赋值
        elif isinstance(node, ast.AnnAssign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            module_level_parts.append(code.strip())
    
    return " ".join(module_level_parts)

# 第308-338行：提取类属性
def _extract_class_attributes(class_node: ast.ClassDef, lines: List[str]) -> str:
    class_attr_parts = []
    
    for node in class_node.body:
        # 提取类属性赋值
        if isinstance(node, ast.Assign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            class_attr_parts.append(code.strip())
        
        # 提取带类型注解的类属性
        elif isinstance(node, ast.AnnAssign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            class_attr_parts.append(code.strip())
    
    return " ".join(class_attr_parts)

# 第545-569行：遍历AST提取函数
for node in tree.body:
    if isinstance(node, ast.ClassDef):
        # 提取类属性
        class_attributes = _extract_class_attributes(node, lines)
        
        # 处理类中的方法
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                process_function(child, node.name, class_attributes, [])
    
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # 处理顶层函数
        process_function(node, None, "", [])
```

**AST提取的信息**:
- ✅ 类定义（`ast.ClassDef`）
- ✅ 函数定义（`ast.FunctionDef`, `ast.AsyncFunctionDef`）
- ✅ Import语句（`ast.Import`, `ast.ImportFrom`）
- ✅ 全局变量赋值（`ast.Assign`, `ast.AnnAssign`）
- ✅ 类属性赋值（类内的`ast.Assign`）
- ✅ 嵌套函数/嵌套类
- ✅ 函数位置和装饰器

**与function_level的区别**:
- `function_level`: 只提取函数，格式为 `# qualified_name\n\n{code}`
- `ir_function`: 提取函数 + 冗余的上下文（file_path + class_name + module_code + class_attrs + function_code）

---

### 间接使用AST的策略（1种）

#### 3. llamaindex_code

**AST使用**: 通过LlamaIndex的`CodeSplitter`类

**代码位置**: `collect_llamaindex_code_blocks(repo_path, ...)` 函数（第1667-1797行）

**实现**:
```python
# 第73-87行：导入LlamaIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter

# 第1705-1711行：创建CodeSplitter
splitter = CodeSplitter(
    language=language,      # "python"
    chunk_lines=chunk_lines,      # 40
    chunk_lines_overlap=chunk_lines_overlap,  # 15
    max_chars=max_chars,      # 1500
)

# 第1713行：调用splitter
nodes = splitter.get_nodes_from_documents(docs)
```

**CodeSplitter的AST使用**:
- ✅ 底层使用AST解析Python代码
- ✅ 识别函数、类、语句
- ✅ 保留代码结构
- ✅ 基于语义边界切分（不是简单行数）
- ❌ 无法看到具体AST操作（封装在库内部）

**注意**: CodeSplitter是一个基于AST的专门切分器，但AST解析过程在库内部完成，不暴露给用户。

---

### 不使用AST的策略（11种）

#### 基础策略（4种）

| 策略 | 切分方式 | 代码位置 |
|-------|---------|---------|
| `fixed` | 按固定行数切分（15行/块） | 第184-194行 |
| `sliding` | 滑动窗口切分（20行窗口） | 第197-208行 |
| `rl_fixed` | 每12个非空行一组 | 第211-224行 |
| `rl_mini` | 按空行分组，每块≤15行 | 第226-257行 |

**特点**:
- ❌ 纯字符串操作：`text.splitlines()`
- ❌ 按行号简单分割
- ❌ 不理解代码语法
- ❌ 可能破坏函数/类边界

---

#### 高级策略（1种）

| 策略 | 切分方式 | 代码位置 |
|-------|---------|---------|
| `epic` | 智能切分（基于token数和字符数） | 第1121-1216行 |

**特点**:
- ⚠️ **不确定是否使用AST**: EpicSplitter是LlamaIndex的切分器，可能使用启发式规则而非AST
- ❌ 基于token数量和字符数动态调整
- ❌ 不直接依赖语法结构

---

#### LangChain策略（3种）

| 策略 | 切分方式 | 代码位置 |
|-------|---------|---------|
| `langchain_fixed` | 按字符数切分（1000字符/块） | 第1445-1540行 |
| `langchain_recursive` | 递归尝试多种分隔符 | 第1335-1429行 |
| `langchain_token` | 按tokenizer token数切分 | 第1556-1651行 |

**特点**:
- ❌ 纯文本处理，不涉及AST
- ❌ 基于字符数或token数分割
- ❌ 使用分隔符规则（`\n\n`, `\n`, `' `, `"`）

---

#### LlamaIndex其他策略（3种）

| 策略 | 切分方式 | 代码位置 |
|-------|---------|---------|
| `llamaindex_sentence` | 按句子切分（基于标点符号） | 第1813-1908行 |
| `llamaindex_token` | 按token数切分 | 第1911-2022行 |
| `llamaindex_semantic` | 基于embedding相似性切分 | 第2038-2142行 |

**特点**:
- ❌ 使用简单的规则或embedding模型
- ❌ 不依赖AST
- ❌ `semantic`策略使用embedding模型，而非AST

---

## 策略对比表

| 策略 | 基于 | AST | 重叠 | Python专用 | 语义感知 | 推荐场景 |
|-------|--------|-----|-------|-----------|----------|----------|
| fixed | 行数 | ❌ | ❌ | ❌ | ❌ | 简单场景 |
| sliding | 行数 | ❌ | ✅ | ❌ | ❌ | 需要上下文 |
| rl_fixed | 论文 | ❌ | ❌ | ❌ | ❌ | 论文复现 |
| rl_mini | 论文 | ❌ | ❌ | ❌ | ❌ | 论文复现 |
| function_level | AST | ✅ | ✅ | ✅ | ❌ | **代码检索** |
| ir_function | AST | ✅ | ❌ | ✅ | ⚠️ | **论文复现** |
| epic | 论文 | ⚠️ | ❌ | ❌ | ❌ | 与BM25一致 |
| langchain_fixed | 字符 | ❌ | ❌ | ❌ | ❌ | 简单文档 |
| langchain_recursive | 字符 | ❌ | ✅ | ❌ | ❌ | 智能文档 |
| langchain_token | Token | ❌ | ❌ | ❌ | ❌ | 精确控制 |
| **llamaindex_code** | **AST** | ✅* | ✅ | ✅ | ❌ | **代码检索** |
| llamaindex_sentence | 句子 | ❌ | ✅ | ❌ | ❌ | 文档检索 |
| llamaindex_token | Token | ❌ | ✅ | ❌ | ❌ | 精确控制 |
| llamaindex_semantic | **语义** | ❌ | ✅ | ❌ | ✅ | **高质量** |

**图例**:
- ✅: 是
- ❌: 否
- ⚠️: 部分支持

---

## 推荐使用

### 代码检索（推荐）

1. **llamaindex_code**: 基于AST，保留代码结构，**最推荐**
2. **function_level**: 函数级语义，简单有效
3. **ir_function**: 论文复现，上下文冗余

### 文档检索

1. **llamaindex_sentence**: 句子结构
2. **langchain_recursive**: 智能切分

### 论文复现

1. **rl_fixed/rl_mini**: RL论文策略
2. **ir_function**: IR论文策略
3. **epic**: BM25一致策略

### 快速原型

1. **fixed**: 最简单
2. **sliding**: 有上下文

---

## Embedding方式

### 统一使用CLS Token

**索引构建** (`batch_build_index.py`):
```python
else:
    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    token_embeddings = outputs[0]
    sent_emb = token_embeddings[:, 0]  # 使用 [CLS] token
    sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
```

**测试查询** (`run_with_index.py`):
```python
outputs = model(input_ids=input_ids, attention_mask=attn_mask)
token_embeddings = outputs[0]
sent_emb = token_embeddings[:, 0]  # 使用 [CLS] token
sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
```

**一致性**: ✅ 索引构建和测试查询都使用CLS token

---

## 使用示例

```bash
# 基础策略
python method/index/batch_build_index.py \
    --repo_path repos/locbench_repos \
    --strategy fixed \
    --model_name models/rlretriever \
    --block_size 15

# 函数级策略
python method/index/batch_build_index.py \
    --repo_path repos/locbench_repos \
    --strategy function_level \
    --model_name models/rlretriever

# IR论文策略
python method/index/batch_build_index.py \
    --repo_path repos/locbench_repos \
    --strategy ir_function \
    --model_name models/rlretriever \
    --ir_function_context_tokens 256

# LlamaIndex CodeSplitter（推荐用于代码）
python method/index/batch_build_index.py \
    --repo_path repos/locbench_repos \
    --strategy llamaindex_code \
    --model_name models/rlretriever \
    --llamaindex_language python \
    --llamaindex_chunk_lines 40 \
    --llamaindex_chunk_lines_overlap 15
```

---

## 注意事项

1. **文件过滤**: 所有策略默认排除 `**/test/**`, `**/tests/**`, `**/test_*.py`, `**/*_test.py`
2. **文件类型**: 默认只处理 `.py`, `.java`, `.js`, `.ts`, `.go`, `.rs`, `.cpp`, `.c`, `.hpp`, `.h`
3. **大文件**: 超过10MB的文件默认跳过（避免内存问题）
4. **空块**: 自动跳过全空块
5. **元数据**: `metadata.jsonl` 包含块的位置、类型、策略等信息

---

## 更新日志

- 2026-01-17: 初始版本，总结14种切块策略
