# IR-Based Baseline 嵌套函数修复总结

## 修复内容

修复了 `ir_function` 策略中嵌套函数的 `qualified_name` 不完整的问题。

## 修复前的问题

**问题**：嵌套函数的 `qualified_name` 缺少父函数路径信息

**示例**：
```python
class Validator:
    def validate(self, data):
        def inner_check(x):  # 嵌套函数
            return x > 0
```

**修复前**：
- `validate`: `helpers.py::Validator::validate` ✅
- `inner_check`: `helpers.py::Validator::inner_check` ❌（缺少父函数信息）

## 修复后的实现

### 主要修改

1. **在 `process_function()` 函数中添加 `function_stack` 参数**（第382行）
2. **使用函数栈追踪嵌套函数的完整路径**（第391行、第451行）
3. **修复 `qualified_name` 构建逻辑**（第409-423行）

### 关键代码变更

**修复前**（第430-433行）：
```python
# 递归处理嵌套函数
for child in ast.iter_child_nodes(node):
    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
        process_function(child, class_name, class_attributes)  # 只传递class_name
```

**修复后**（第446-451行）：
```python
# 递归处理嵌套函数（传递函数栈的副本）
for child in ast.iter_child_nodes(node):
    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
        process_function(child, class_name, class_attributes, function_stack.copy())  # 传递函数栈
```

**修复后的 `qualified_name` 构建**（第409-423行）：
```python
# 构建完全限定名（用于 metadata）- 考虑嵌套函数
if class_name:
    if len(function_stack) > 1:
        # 嵌套函数：使用完整的函数路径
        qualified_name = f"{rel}::{class_name}::{'::'.join(function_stack)}"
    else:
        # 顶层方法
        qualified_name = f"{rel}::{class_name}::{function_name}"
else:
    if len(function_stack) > 1:
        # 嵌套函数：使用完整的函数路径
        qualified_name = f"{rel}::{'::'.join(function_stack)}"
    else:
        # 顶层函数
        qualified_name = f"{rel}::{function_name}"
```

## 修复后的结果

**示例**：
```python
class Validator:
    def validate(self, data):
        def inner_check(x):
            def deep_check(y):  # 深度嵌套
                return y > 0
            return deep_check(x)
        return all(inner_check(d) for d in data)
```

**修复后**：
- `validate`: `helpers.py::Validator::validate` ✅
- `inner_check`: `helpers.py::Validator::validate::inner_check` ✅
- `deep_check`: `helpers.py::Validator::validate::inner_check::deep_check` ✅

**顶层函数示例**：
```python
def top_function():
    def nested_in_top():
        return 42
```

**修复后**：
- `top_function`: `helpers.py::top_function` ✅
- `nested_in_top`: `helpers.py::top_function::nested_in_top` ✅

## 修改的文件

- `IR-based/method/index/batch_build_index.py`
  - `blocks_ir_function()` 函数中的 `process_function()` 内部函数（第378-451行）
  - 顶层调用处（第462行、第468-473行、第477行）

## 测试验证

修复后的代码能够：
1. ✅ 正确提取嵌套函数
2. ✅ 为嵌套函数构建完整的 `qualified_name`
3. ✅ 处理多层嵌套（如 `validate::inner_check::deep_check`）
4. ✅ 同时支持类方法和顶层函数的嵌套

## 与论文要求的一致性

修复后的实现现在**完全符合**论文要求：

1. ✅ **函数级索引粒度**：每个函数作为独立文档
2. ✅ **上下文增强**：模块级代码和类属性附加到函数表示
3. ✅ **扁平化索引**：所有函数平铺到统一列表
4. ✅ **递归AST解析**：正确提取嵌套函数
5. ✅ **完整的函数路径**：嵌套函数的 `qualified_name` 包含完整的父函数路径

## 注意事项

- 函数栈使用 `copy()` 传递，确保每个递归调用有独立的栈副本
- 函数退出时使用 `pop()` 清理栈，保持栈的正确性
- 嵌套类中的方法也从空栈开始（第462行、第472行），确保类方法之间的独立性
- 语法检查通过，代码可以正常运行
