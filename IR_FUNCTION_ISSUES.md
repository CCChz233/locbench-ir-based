# IR-Based Baseline 实现问题分析

## 关键问题发现

经过详细检查，发现 `ir_function` 策略在**嵌套函数处理**方面存在一个关键问题。

---

## 问题：嵌套函数的 Qualified Name 不完整

### 问题描述

**位置**: `blocks_ir_function()` 函数中的 `process_function()` (第378-433行)

**当前实现**（第430-433行）：
```python
# 递归处理嵌套函数
for child in ast.iter_child_nodes(node):
    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
        process_function(child, class_name, class_attributes)
```

**问题**：
- 嵌套函数调用 `process_function(child, class_name, class_attributes)`
- 嵌套函数使用**父函数的 `class_name`**，但没有包含**父函数的名称**
- 导致嵌套函数的 `qualified_name` 不完整

### 示例

**代码**：
```python
# helpers.py
class Validator:
    def validate(self, data):
        def inner_check(x):  # 嵌套函数
            return x > 0
        return all(inner_check(d) for d in data)
```

**论文期望的 qualified_name**：
- `validate`: `helpers.py::Validator::validate`
- `inner_check`: `helpers.py::Validator::validate.<locals>.inner_check` 或类似格式

**当前实现产生的 qualified_name**：
- `validate`: `helpers.py::Validator::validate` ✅
- `inner_check`: `helpers.py::Validator::inner_check` ❌（缺少 `validate.<locals>` 部分）

### 代码位置

第405-407行（构建qualified_name）：
```python
if class_name:
    qualified_name = f"{rel}::{class_name}::{function_name}"
else:
    qualified_name = f"{rel}::{function_name}"
```

嵌套函数在递归调用时（第433行）：
```python
process_function(child, class_name, class_attributes)  # class_name是父函数的类名
```

嵌套函数无法知道父函数的名称，因此无法构建完整的路径。

---

## 对比：function_level 策略的实现

**位置**: `blocks_by_function()` 函数（第697-903行）

**function_level 策略使用了函数栈**（第728-729行）：
```python
function_stack: List[str] = []  # 用于嵌套函数
```

**但是在 `ir_function` 策略中没有使用函数栈**！

---

## 其他检查结果

### ✅ 符合论文要求的部分

1. **函数级索引粒度** ✅
   - 每个函数作为独立的Block（第419行）
   
2. **上下文增强** ✅
   - 模块级代码被提取并附加（第309-310行）
   - 类属性被提取并附加（第313-314行）
   
3. **扁平化索引** ✅
   - 所有函数平铺到一个列表（`collect_ir_function_blocks` 第548行）

4. **AST解析** ✅
   - 使用 `ast.parse()` 解析代码（第357行）
   
5. **递归提取嵌套函数** ✅
   - 使用 `ast.iter_child_nodes()` 递归访问（第431行）
   - **嵌套函数确实会被提取**

### ⚠️ 格式问题（次要）

**上下文格式**（第319行）：
```python
return " ".join(parts)  # 所有内容用空格连接
```

- 所有内容（file_path, class_name, module_code, class_attrs, function_code）用空格连接成一行
- 论文示例显示有多行格式，但代码中用空格连接
- **影响**：可能不影响embedding效果，但格式不匹配

---

## 建议修复方案

### 方案1：引入函数栈（推荐）

修改 `process_function()` 函数，添加函数栈参数：

```python
def process_function(
    node,
    class_name: Optional[str] = None,
    class_attributes: str = "",
    function_stack: List[str] = None,  # 新增
):
    """处理单个函数/方法"""
    if function_stack is None:
        function_stack = []
    
    function_name = node.name
    function_stack.append(function_name)  # 添加到栈
    
    # 构建qualified_name
    if class_name:
        if function_stack:
            # 嵌套函数：helpers.py::Class::outer.<locals>.inner
            path_parts = [rel, class_name] + function_stack
            qualified_name = "::".join(path_parts)
        else:
            qualified_name = f"{rel}::{class_name}::{function_name}"
    else:
        if function_stack:
            path_parts = [rel] + function_stack
            qualified_name = "::".join(path_parts)
        else:
            qualified_name = f"{rel}::{function_name}"
    
    # ... 创建Block ...
    
    # 递归处理嵌套函数
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            process_function(child, class_name, class_attributes, function_stack.copy())  # 传递栈的副本
    
    function_stack.pop()  # 退出时弹出
```

### 方案2：使用函数路径字符串

```python
def process_function(
    node,
    class_name: Optional[str] = None,
    class_attributes: str = "",
    parent_function_path: str = "",  # 新增：父函数的路径
):
    function_name = node.name
    
    # 构建qualified_name
    if parent_function_path:
        if class_name:
            qualified_name = f"{rel}::{class_name}::{parent_function_path}.<locals>.{function_name}"
        else:
            qualified_name = f"{rel}::{parent_function_path}.<locals>.{function_name}"
    else:
        if class_name:
            qualified_name = f"{rel}::{class_name}::{function_name}"
        else:
            qualified_name = f"{rel}::{function_name}"
    
    # 递归处理嵌套函数
    current_path = function_name if not parent_function_path else f"{parent_function_path}.<locals>.{function_name}"
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            process_function(child, class_name, class_attributes, current_path)
```

---

## 总结

### 关键问题

1. ❌ **嵌套函数的qualified_name不完整**：缺少父函数路径信息
2. ⚠️ **上下文格式**：使用空格连接而非多行格式（可能不影响功能）

### 符合要求的部分

1. ✅ 函数级索引粒度
2. ✅ 上下文增强（模块级代码和类属性）
3. ✅ 扁平化索引
4. ✅ AST解析和递归提取嵌套函数

### 优先级

- **高优先级**：修复嵌套函数的qualified_name（如果论文要求完整的路径）
- **低优先级**：上下文格式（如果embedding模型能处理空格连接的文本，可能不需要修改）
