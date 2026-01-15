# 代码分块（Chunking）策略说明

IR-based 框架支持 **4种分块策略**，用于将代码文件切分为代码块（blocks），然后构建索引进行检索。

## 4种分块策略

### 1. `fixed` - 固定行切块（不重叠）

**特点**：按固定行数切分，块之间不重叠

**实现函数**：`blocks_fixed_lines()`

**算法**：
- 从文件开头开始，每 `block_size` 行切一块
- 跳过空块（全为空行）
- 块之间不重叠

**参数**：
- `--block_size`: 每块的行数（默认：15）

**示例**：
```python
# 文件有 50 行代码
# block_size = 15
# 结果：块1(0-14行), 块2(15-29行), 块3(30-44行), 块4(45-49行)
```

**使用命令**：
```bash
python method/index/batch_build_index.py \
    --strategy fixed \
    --block_size 15 \
    --repo_path ../repos/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever
```

**适用场景**：简单高效，适合快速索引构建

---

### 2. `sliding` - 滑动窗口切块（可重叠）

**特点**：使用滑动窗口，块之间可以重叠

**实现函数**：`blocks_sliding()`

**算法**：
- 窗口大小：`window_size`（默认：20行）
- 步长：`window_size // slice_size`（默认：20 // 2 = 10行）
- 每个窗口中心位置向前后各扩展 `delta = window_size // 2`
- 块之间可以重叠，提供更多上下文

**参数**：
- `--window_size`: 窗口大小（默认：20）
- `--slice_size`: 切片大小，控制步长（默认：2）

**示例**：
```python
# window_size = 20, slice_size = 2
# 步长 = 20 // 2 = 10
# 结果：块1(0-19行), 块2(5-24行), 块3(10-29行), ...
```

**使用命令**：
```bash
python method/index/batch_build_index.py \
    --strategy sliding \
    --window_size 20 \
    --slice_size 2 \
    --repo_path ../repos/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever
```

**适用场景**：需要更多上下文信息，提高检索召回率

---

### 3. `rl_fixed` - RLCoder 固定块

**特点**：按非空行数切分，每块12个非空行

**实现函数**：`blocks_rl_fixed()`

**算法**：
- 先过滤掉空行，只保留非空行
- 每12个非空行为一块
- 最多处理 `max_lines` 行（默认：5000行）

**参数**：
- `max_lines`: 最大处理行数（默认：5000，代码内部参数）

**示例**：
```python
# 文件有：空行、代码行、空行、代码行...
# 过滤后：30个非空行
# 结果：块1(0-11非空行), 块2(12-23非空行), 块3(24-29非空行)
```

**使用命令**：
```bash
python method/index/batch_build_index.py \
    --strategy rl_fixed \
    --repo_path ../repos/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever
```

**适用场景**：与 RLCoder 方法兼容，忽略空行影响

---

### 4. `rl_mini` - RLCoder Mini块

**特点**：按空行分段，然后拼接成≤15行的块

**实现函数**：`blocks_rl_mini()`

**算法**：
1. 按空行分段，每个连续非空行段为一个 mini_block
2. 如果 mini_block 超过15行，按15行切分
3. 将多个 mini_block 拼接，每个最终块不超过15行
4. 最多处理 `max_lines` 行（默认：5000行）

**参数**：
- `max_lines`: 最大处理行数（默认：5000，代码内部参数）

**示例**：
```python
# 文件结构：
# 行1-5: 代码（mini_block1）
# 空行
# 行7-12: 代码（mini_block2）
# 空行
# 行14-20: 代码（mini_block3）
# 
# 结果：块1(mini_block1 + mini_block2 = 11行), 块2(mini_block3 = 7行)
```

**使用命令**：
```bash
python method/index/batch_build_index.py \
    --strategy rl_mini \
    --repo_path ../repos/locbench_repos \
    --index_dir index_data \
    --model_name models/rlretriever
```

**适用场景**：尊重代码的自然结构（函数/类边界），提供语义连贯的块

---

## 策略对比

| 策略 | 重叠 | 块大小 | 是否考虑空行 | 优势 | 劣势 |
|------|------|--------|--------------|------|------|
| `fixed` | ❌ | 固定（15行） | 不考虑 | 简单快速，均匀切分 | 可能切分代码逻辑 |
| `sliding` | ✅ | 固定（20行） | 不考虑 | 提供更多上下文 | 索引更大，可能有冗余 |
| `rl_fixed` | ❌ | 固定（12非空行） | 考虑（忽略空行） | 忽略格式化差异 | 块大小可能不一致 |
| `rl_mini` | ❌ | 可变（≤15行） | 考虑（按空行分段） | 尊重代码结构 | 算法较复杂 |

## 代码实现位置

所有分块策略的实现都在：
- **文件**：`method/index/build_index.py`
- **函数**：
  - `blocks_fixed_lines()` - fixed策略
  - `blocks_sliding()` - sliding策略
  - `blocks_rl_fixed()` - rl_fixed策略
  - `blocks_rl_mini()` - rl_mini策略
- **调用入口**：`collect_blocks()` 函数根据策略参数选择对应的分块函数

## 索引数据目录

构建后的索引保存在 `index_data/` 目录下，常见命名模式：

- `dense_index_fixed/` - fixed策略构建的索引
- `dense_index_sliding/` - sliding策略构建的索引
- `dense_index_rl_fixed/` - rl_fixed策略构建的索引
- `dense_index_rl_mini/` - rl_mini策略构建的索引

## 选择建议

1. **快速测试/基线**：使用 `fixed`，简单高效
2. **追求召回率**：使用 `sliding`，提供更多重叠上下文
3. **与RLCoder兼容**：使用 `rl_fixed` 或 `rl_mini`
4. **尊重代码结构**：使用 `rl_mini`，按空行分段更符合代码语义
