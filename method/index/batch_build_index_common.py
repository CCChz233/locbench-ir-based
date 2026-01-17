#!/usr/bin/env python
"""
共享的索引构建代码（不包含模型相关逻辑）

包含：
- Block 类定义
- 所有切块策略函数
- BlockDataset 类
- 工具函数（save_index, iter_files 等）
"""

import ast
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import torch
from torch.utils.data import Dataset
import logging

# ============================================================================
# Block 类和配置
# ============================================================================


class Block:
    def __init__(
        self,
        file_path: str,
        start: int,
        end: int,
        content: str,
        block_type: str,
        context_text: Optional[str] = None,
        function_text: Optional[str] = None,
    ):
        self.file_path = file_path
        self.start = start
        self.end = end
        self.content = content
        self.block_type = block_type
        self.context_text = context_text
        self.function_text = function_text


@dataclass
class EpicSplitterConfig:
    """EpicSplitter 配置，尽量与 BM25 保持一致。"""
    min_chunk_size: int = 512
    chunk_size: int = 1024
    max_chunk_size: int = 2048
    hard_token_limit: int = 4096
    max_chunks: int = 512

    @classmethod
    def from_bm25_config(cls):
        """
        尝试从 BM25 配置读取参数；若不可用则使用本地默认值。
        """
        try:
            # 目前 BM25 构建未暴露默认常量，这里先使用本地默认值。
            # 若后续在 bm25_retriever 中新增常量，可在此导入以保持完全一致。
            return cls()
        except Exception:
            return cls()


# ============================================================================
# 基础切块策略
# ============================================================================


def iter_files(repo_root: Path) -> List[Path]:
    return [
        p for p in repo_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".py", ".java", ".js", ".ts", ".go", ".rs", ".cpp", ".c", ".hpp", ".h"}
    ]


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


def blocks_sliding(text: str, rel: str, window_size: int, slice_size: int) -> List[Block]:
    lines = text.splitlines()
    delta = window_size // 2
    step = max(1, window_size // slice_size) if slice_size > 0 else window_size
    blocks: List[Block] = []
    for line_no in range(0, len(lines), step):
        start = max(0, line_no - delta)
        end = min(len(lines), line_no + window_size - delta)
        chunk = lines[start:end]
        if any(l.strip() for l in chunk):
            blocks.append(Block(rel, start, end - 1, "\n".join(chunk), "sliding"))
    return blocks


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


# ============================================================================
# IR-based 函数级分块（论文复现）
# ============================================================================


def _extract_module_level_code(tree: ast.AST, lines: List[str]) -> str:
    """
    提取模块级代码：imports, 全局变量, 顶层赋值等

    按照论文：这些代码会被冗余复制到该文件每个函数的表示中

    Args:
        tree: AST 树
        lines: 源代码行列表

    Returns:
        模块级代码字符串（多行合并为单行，空格分隔）
    """
    module_level_parts = []

    for node in tree.body:
        # import 语句
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])  # 合并为单行
            module_level_parts.append(code.strip())

        # 全局变量赋值
        elif isinstance(node, ast.Assign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            module_level_parts.append(code.strip())

        # 带类型注解的赋值 (e.g., DEBUG: bool = True)
        elif isinstance(node, ast.AnnAssign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            module_level_parts.append(code.strip())

    return " ".join(module_level_parts)


def _extract_class_attributes(class_node: ast.ClassDef, lines: List[str]) -> str:
    """
    提取类属性（非方法的类级别定义）

    按照论文：类属性会被冗余复制到该类每个方法的表示中

    Args:
        class_node: 类的 AST 节点
        lines: 源代码行列表

    Returns:
        类属性代码字符串（多行合并为单行，空格分隔）
    """
    class_attr_parts = []

    for node in class_node.body:
        # 类属性赋值
        if isinstance(node, ast.Assign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            class_attr_parts.append(code.strip())

        # 带类型注解的类属性
        elif isinstance(node, ast.AnnAssign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            class_attr_parts.append(code.strip())

    return " ".join(class_attr_parts)


def _build_ir_function_representation(
    file_path: str,
    class_name: Optional[str],
    function_code: str,
    module_level_code: str,
    class_attributes: str,
) -> str:
    """
    按照论文 IR-based 方法构建函数表示

    论文原文 (Appendix C.1.1):
    "The function's context—its containing file and class—is appended to
    the function representation before embedding"

    格式: {file_path} {class_name} {module_level_code} {class_attributes} {function_code}

    Args:
        file_path: 文件路径
        class_name: 类名（如果是方法）
        function_code: 函数源代码
        module_level_code: 模块级代码（imports, 全局变量等）
        class_attributes: 类属性代码（如果是方法）

    Returns:
        完整的函数表示，用于 embedding
    """
    context_text = _build_ir_function_context(
        file_path=file_path,
        class_name=class_name,
        module_level_code=module_level_code,
        class_attributes=class_attributes,
    )
    if context_text.strip():
        return f"{context_text} {function_code}"
    return function_code


def _build_ir_function_context(
    file_path: str,
    class_name: Optional[str],
    module_level_code: str,
    class_attributes: str,
) -> str:
    parts = [file_path]

    if class_name:
        parts.append(class_name)

    # 添加模块级上下文（冗余复制到每个函数）
    if module_level_code.strip():
        parts.append(module_level_code.strip())

    # 添加类属性（如果是类方法）
    if class_attributes.strip():
        parts.append(class_attributes.strip())

    return " ".join(parts)


def blocks_ir_function(text: str, rel: str) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    """
    按照论文 IR-based 方法提取函数级代码块（Paper Reproduction）

    论文关键特点（Appendix C.1.1）：
    1. 每个函数单独作为一个嵌入单元（embedding unit）
    2. 模块级代码（imports, 全局变量）冗余复制到该文件每个函数
    3. 类属性冗余复制到该类每个方法
    4. 使用 Flat Indexing（无层级结构，无单独的全局变量索引）
    5. 非 Python 文件不处理（只索引函数）

    与我们的 function_level 的区别：
    - function_level: 只有 qualified_name + 函数代码
    - ir_function: file_path + class_name + module_code + class_attrs + 函数代码

    Args:
        text: 文件内容
        rel: 相对文件路径

    Returns:
        (blocks, function_metadata): 代码块列表和函数元数据字典
    """
    blocks: List[Block] = []
    function_metadata: Dict[int, Dict[str, Optional[str]]] = {}

    # 只处理 Python 文件（论文只在函数级别索引）
    if not rel.endswith('.py'):
        return blocks, function_metadata

    # 检查文件大小
    if len(text) > 10 * 1024 * 1024:  # 10MB
        logging.debug(f"File {rel} is too large, skipping")
        return blocks, function_metadata

    try:
        tree = ast.parse(text)
        lines = text.splitlines()

        # 1. 提取模块级代码（将冗余复制到每个函数）
        module_level_code = _extract_module_level_code(tree, lines)

        def get_function_code(node) -> str:
            """获取函数的完整代码（包括装饰器）"""
            func_start = node.lineno - 1
            func_end = node.end_lineno if hasattr(node, 'end_lineno') else func_start + 10

            # 获取装饰器起始行
            decorator_start = func_start
            if node.decorator_list:
                first_decorator = node.decorator_list[0]
                if hasattr(first_decorator, 'lineno'):
                    decorator_start = first_decorator.lineno - 1

            all_lines = lines[decorator_start:func_end]
            return "\n".join(all_lines)

        def process_function(
            node,
            class_name: Optional[str] = None,
            class_attributes: str = "",
            function_stack: List[str] = None,
        ):
            """处理单个函数/方法"""
            nonlocal blocks, function_metadata

            if function_stack is None:
                function_stack = []

            function_name = node.name
            function_stack.append(function_name)  # 添加到栈

            # 获取行号范围
            start_line = node.lineno - 1
            end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else start_line + 10

            # 包含装饰器的起始行
            if node.decorator_list:
                first_decorator = node.decorator_list[0]
                if hasattr(first_decorator, 'lineno'):
                    start_line = first_decorator.lineno - 1

            # 获取函数代码
            func_code = get_function_code(node)
            if not func_code.strip():
                function_stack.pop()  # 退出时弹出
                return

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

            # 按照论文方式构建 IR 表示
            context_text = _build_ir_function_context(
                file_path=rel,
                class_name=class_name,
                module_level_code=module_level_code,
                class_attributes=class_attributes,
            )
            ir_representation = _build_ir_function_representation(
                file_path=rel,
                class_name=class_name,
                function_code=func_code,
                module_level_code=module_level_code,
                class_attributes=class_attributes,
            )

            # 创建 Block
            block = Block(
                rel,
                start_line,
                end_line,
                ir_representation,
                "ir_function",
                context_text=context_text,
                function_text=func_code,
            )
            block_index = len(blocks)
            blocks.append(block)

            # 保存元数据
            function_metadata[block_index] = {
                "qualified_name": qualified_name,
                "class_name": class_name,
                "function_name": function_name,
            }

            # 递归处理嵌套函数（传递函数栈的副本）
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    process_function(child, class_name, class_attributes, function_stack.copy())

            function_stack.pop()  # 退出时弹出

        # 2. 遍历 AST 顶层节点
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                # 提取类属性（将冗余复制到该类每个方法）
                class_attributes = _extract_class_attributes(node, lines)

                # 处理类中的方法
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        process_function(child, node.name, class_attributes, [])
                    # 处理嵌套类
                    elif isinstance(child, ast.ClassDef):
                        nested_class_attrs = _extract_class_attributes(child, lines)
                        for nested_child in child.body:
                            if isinstance(nested_child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                process_function(
                                    nested_child,
                                    f"{node.name}.{child.name}",
                                    nested_class_attrs,
                                    []  # 嵌套类的方法也使用空栈开始
                                )

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 顶层函数（没有类属性）
                process_function(node, None, "", [])

    except SyntaxError as e:
        logging.debug(f"Syntax error in {rel} (line {e.lineno}): {e.msg}. Skipping.")
        return blocks, function_metadata
    except Exception as e:
        logging.error(f"Error processing {rel}: {type(e).__name__}: {e}. Skipping.")
        return blocks, function_metadata

    return blocks, function_metadata


def collect_ir_function_blocks(repo_path: str) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    """
    收集 IR-based 函数级代码块（论文复现版本）

    按照论文 Appendix C.1.1：
    - 每个函数单独作为一个嵌入单元
    - 模块级代码冗余复制到该文件每个函数
    - 类属性冗余复制到该类每个方法
    - 只索引 Python 文件中的函数

    Args:
        repo_path: 仓库路径

    Returns:
        (blocks, function_metadata): 代码块列表和函数元数据字典
    """
    logger = logging.getLogger(__name__)
    repo_root = Path(repo_path)
    all_blocks: List[Block] = []
    all_function_metadata: Dict[int, Dict[str, Optional[str]]] = {}

    # 只处理 Python 文件
    python_files = [p for p in iter_files(repo_root) if p.suffix.lower() == '.py']
    total_files = len(python_files)

    if total_files == 0:
        logger.debug(f"No Python files found in {repo_path}")
        return all_blocks, all_function_metadata

    logger.info(f"Processing {total_files} Python files in {repo_path} (IR-based paper style)")

    stats = {
        'total_files': total_files,
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'total_functions': 0,
        'files_with_functions': 0,
    }

    for p in python_files:
        stats['processed'] += 1

        # 检查文件大小
        try:
            file_size = p.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB
                logger.debug(f"Skipping large file {p.name}")
                stats['skipped'] += 1
                continue
        except Exception:
            stats['skipped'] += 1
            continue

        # 读取文件
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                stats['skipped'] += 1
                continue
        except Exception:
            stats['skipped'] += 1
            continue

        rel = str(p.relative_to(repo_root))

        # 使用 IR-based 方法提取
        file_blocks, file_metadata = blocks_ir_function(text, rel)

        # 调整 metadata 索引
        current_start_idx = len(all_blocks)
        for local_idx, metadata in file_metadata.items():
            global_idx = current_start_idx + local_idx
            all_function_metadata[global_idx] = metadata

        all_blocks.extend(file_blocks)

        if file_blocks:
            stats['files_with_functions'] += 1
            stats['total_functions'] += len(file_blocks)

    logger.info(
        f"Collected {len(all_blocks)} IR-style function blocks from {repo_path} "
        f"(files: {stats['files_with_functions']}, "
        f"avg per file: {stats['total_functions'] / max(stats['files_with_functions'], 1):.1f})"
    )

    return all_blocks, all_function_metadata


# ============================================================================
# 原有的 function_level 实现（保留兼容）
# ============================================================================


def build_function_context_content(
    file_path: str,
    class_name: Optional[str],
    function_name: str,
    qualified_name: str,
    original_code: str,
    start_line: int,  # 保留但不在输出中使用
    end_line: int,    # 保留但不在输出中使用
) -> str:
    """
    构建超简洁的上下文（论文风格）

    格式: # qualified_name\n\n{code}
    示例: # src/utils.py::MathUtils::calculate_sum\n\ndef calculate_sum(a, b):\n    return a + b

    Args:
        file_path: 文件路径（保留但不在输出中使用）
        class_name: 类名（保留但不在输出中使用）
        function_name: 函数名（保留但不在输出中使用）
        qualified_name: 完全限定名（用于注释）
        original_code: 原始函数代码
        start_line: 起始行号（保留但不在输出中使用）
        end_line: 结束行号（保留但不在输出中使用）

    Returns:
        增强后的代码内容（只包含一行注释标识符和代码）
    """
    # 只添加一行注释作为标识符
    header = f"# {qualified_name}\n\n"
    return header + original_code


def blocks_function_level_with_fallback(
    text: str,
    rel: str,
    block_size: int = 15
) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    """
    函数级切块 + 未覆盖代码的 fixed 切块作为补充

    解决 function_level 策略只索引函数、遗漏大量非函数代码的问题：
    - 模块级代码（全局变量、常量定义、import 语句）
    - 类定义本身（类属性、类级代码）
    - 脚本入口代码（if __name__ == "__main__": 下的非函数代码）

    Args:
        text: 文件内容
        rel: 相对文件路径
        block_size: 对未覆盖代码使用的 fixed 切块大小，默认 15 行

    Returns:
        (blocks, function_metadata): 代码块列表和函数元数据字典
    """
    # 1. 先提取所有函数级块
    function_blocks, function_metadata = blocks_by_function(text, rel)
    function_metadata_by_block = {
        function_blocks[idx]: metadata for idx, metadata in function_metadata.items()
    }

    # 非 Python 文件或解析失败时，直接使用 fixed 策略
    if not rel.endswith('.py') or not function_blocks:
        fixed_blocks = blocks_fixed_lines(text, rel, block_size)
        # 修改 block_type 以区分
        for b in fixed_blocks:
            b.block_type = "function_level_fallback"
        return fixed_blocks, {}

    # 2. 记录被函数覆盖的行号
    lines = text.splitlines()
    total_lines = len(lines)
    covered_lines = set()

    for block in function_blocks:
        for line_no in range(block.start, block.end + 1):
            covered_lines.add(line_no)

    # 3. 找出未被覆盖的连续行区间
    uncovered_ranges = []
    current_start = None

    for i in range(total_lines):
        if i not in covered_lines:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                uncovered_ranges.append((current_start, i - 1))
                current_start = None

    # 处理末尾的未覆盖区间
    if current_start is not None:
        uncovered_ranges.append((current_start, total_lines - 1))

    # 4. 对未覆盖的区间使用 fixed 策略切分
    fallback_blocks = []
    for range_start, range_end in uncovered_ranges:
        # 提取该区间的文本
        range_lines = lines[range_start:range_end + 1]
        range_text = "\n".join(range_lines)

        # 跳过全空区间
        if not any(l.strip() for l in range_lines):
            continue

        # 对该区间使用 fixed 切块
        chunk_start = 0
        while chunk_start < len(range_lines):
            chunk_end = min(chunk_start + block_size, len(range_lines))
            chunk_lines = range_lines[chunk_start:chunk_end]

            # 跳过全空块
            if any(l.strip() for l in chunk_lines):
                # 计算在原文件中的真实行号
                real_start = range_start + chunk_start
                real_end = range_start + chunk_end - 1
                content = "\n".join(chunk_lines)

                fallback_blocks.append(Block(
                    rel, real_start, real_end, content, "function_level_fallback"
                ))

            chunk_start = chunk_end

    # 5. 合并函数块和补充块
    all_blocks = function_blocks + fallback_blocks

    # 按起始行号排序，保持文件顺序
    all_blocks.sort(key=lambda b: b.start)

    sorted_metadata = {}
    for idx, block in enumerate(all_blocks):
        metadata = function_metadata_by_block.get(block)
        if metadata is not None:
            sorted_metadata[idx] = metadata

    return all_blocks, sorted_metadata


def blocks_by_function(text: str, rel: str) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    """
    使用AST解析提取所有函数和类方法

    Args:
        text: 文件内容
        rel: 相对文件路径

    Returns:
        (blocks, function_metadata): 代码块列表和函数元数据字典

    注意：如果AST解析失败，返回空blocks和空metadata，不回退到其他策略
    这样可以保持策略的一致性（block_type始终是"function_level"或为空）
    """
    blocks: List[Block] = []
    function_metadata: Dict[int, Dict[str, Optional[str]]] = {}

    # 只处理Python文件
    if not rel.endswith('.py'):
        return blocks, function_metadata

    # 检查文件大小，避免处理过大的文件
    if len(text) > 10 * 1024 * 1024:  # 10MB
        logging.debug(f"File {rel} is too large ({len(text) / 1024 / 1024:.2f}MB), skipping")
        return blocks, function_metadata

    try:
        tree = ast.parse(text)
        lines = text.splitlines()

        # 维护类栈和函数栈来追踪嵌套结构
        class_stack: List[str] = []  # 只保存直接父类（用于嵌套类）
        function_stack: List[str] = []  # 用于嵌套函数

        def get_function_code(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
            """获取函数的完整代码（包括装饰器）"""
            # 获取函数体的行号范围
            func_start = node.lineno - 1  # AST行号从1开始，转换为0-based
            func_end = node.end_lineno - 1 if hasattr(node, 'end_lineno') else func_start + 10

            # 获取装饰器：装饰器通常在函数定义之前，需要从函数定义行向上查找
            decorator_start = func_start
            if node.decorator_list:
                # 找到第一个装饰器的起始行
                first_decorator = node.decorator_list[0]
                if hasattr(first_decorator, 'lineno'):
                    decorator_start = first_decorator.lineno - 1

            # 获取从第一个装饰器到函数结束的所有行
            all_lines = lines[decorator_start:func_end + 1]
            return "\n".join(all_lines)

        def visit_node(node: ast.AST):
            """递归访问AST节点"""
            nonlocal blocks, function_metadata

            # 处理类定义
            if isinstance(node, ast.ClassDef):
                # 只保存直接父类（不包含外层类）
                class_stack.append(node.name)

                # 访问类体中的节点
                for child in node.body:
                    visit_node(child)

                # 弹出当前类
                class_stack.pop()

            # 处理函数定义
            elif isinstance(node, ast.FunctionDef):
                function_name = node.name
                function_stack.append(function_name)

                # 获取行号
                start_line = node.lineno - 1  # 转换为0-based
                end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else start_line + 10
                if node.decorator_list:
                    first_decorator = node.decorator_list[0]
                    if hasattr(first_decorator, 'lineno'):
                        start_line = min(start_line, first_decorator.lineno - 1)

                # 获取函数完整代码
                func_code = get_function_code(node)

                if not func_code.strip():
                    function_stack.pop()
                    # 继续访问嵌套函数
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            visit_node(child)
                    return

                # 构建完全限定名
                if class_stack:
                    # 类方法：只包含直接父类
                    class_name = class_stack[-1]
                    qualified_name = f"{rel}::{class_name}::{function_name}"
                else:
                    # 顶层函数
                    qualified_name = f"{rel}::{function_name}"

                # 构建上下文增强的代码
                enhanced_content = build_function_context_content(
                    file_path=rel,
                    class_name=class_stack[-1] if class_stack else None,
                    function_name=function_name,
                    qualified_name=qualified_name,
                    original_code=func_code,
                    start_line=start_line,
                    end_line=end_line,
                )

                # 创建Block对象
                block = Block(rel, start_line, end_line, enhanced_content, "function_level")
                block_index = len(blocks)
                blocks.append(block)

                # 保存元数据
                function_metadata[block_index] = {
                    "qualified_name": qualified_name,
                    "class_name": class_stack[-1] if class_stack else None,
                    "function_name": function_name,
                }

                # 访问嵌套函数（函数内部定义的函数）
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        visit_node(child)

                function_stack.pop()

            # 处理异步函数（与普通函数处理逻辑相同，可以合并）
            elif isinstance(node, ast.AsyncFunctionDef):
                # 异步函数处理逻辑与普通函数相同
                function_name = node.name
                function_stack.append(function_name)

                start_line = node.lineno - 1
                end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else start_line + 10
                if node.decorator_list:
                    first_decorator = node.decorator_list[0]
                    if hasattr(first_decorator, 'lineno'):
                        start_line = min(start_line, first_decorator.lineno - 1)

                # 使用get_function_code获取完整代码（包括装饰器）
                func_code = get_function_code(node)

                if not func_code.strip():
                    function_stack.pop()
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            visit_node(child)
                    return

                if class_stack:
                    class_name = class_stack[-1]
                    qualified_name = f"{rel}::{class_name}::{function_name}"
                else:
                    qualified_name = f"{rel}::{function_name}"

                enhanced_content = build_function_context_content(
                    file_path=rel,
                    class_name=class_stack[-1] if class_stack else None,
                    function_name=function_name,
                    qualified_name=qualified_name,
                    original_code=func_code,
                    start_line=start_line,
                    end_line=end_line,
                )

                block = Block(rel, start_line, end_line, enhanced_content, "function_level")
                block_index = len(blocks)
                blocks.append(block)

                function_metadata[block_index] = {
                    "qualified_name": qualified_name,
                    "class_name": class_stack[-1] if class_stack else None,
                    "function_name": function_name,
                }

                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        visit_node(child)

                function_stack.pop()

            # 对于其他节点，继续访问子节点
            else:
                if hasattr(node, 'body'):
                    for child in node.body:
                        visit_node(child)

        # 从顶层节点开始访问
        for node in tree.body:
            visit_node(node)

    except SyntaxError as e:
        # 语法错误：文件可能不是有效的Python代码，跳过该文件
        # 使用debug级别，因为可能是正常的（如.py文件包含其他内容）
        logging.debug(f"Syntax error in {rel} (line {e.lineno}): {e.msg}. Skipping file.")
        return blocks, function_metadata
    except (ValueError, AttributeError) as e:
        # AST相关错误：可能是Python版本不兼容或其他问题
        logging.warning(f"AST parsing error in {rel}: {e}. Skipping file.")
        return blocks, function_metadata
    except MemoryError as e:
        # 内存错误：文件太大或AST解析消耗过多内存
        logging.error(f"Memory error processing {rel}: {e}. Skipping file.")
        return blocks, function_metadata
    except Exception as e:
        # 其他未知错误：记录详细信息并跳过
        logging.error(f"Unexpected error processing {rel}: {type(e).__name__}: {e}. Skipping file.")
        return blocks, function_metadata

    return blocks, function_metadata


def collect_function_blocks(repo_path: str, block_size: int = 15) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    """
    收集函数级代码块（带 fallback），返回blocks和function_metadata

    改进：使用 blocks_function_level_with_fallback，确保完整覆盖所有代码：
    - 函数和方法使用 function_level 切块
    - 未被函数覆盖的代码（imports、全局变量、类属性等）使用 fixed 切块补充

    Args:
        repo_path: 仓库路径
        block_size: 对未覆盖代码使用的 fixed 切块大小，默认 15 行

    Returns:
        (blocks, function_metadata): 代码块列表和函数元数据字典

    注意：如果整个仓库没有提取到任何blocks，返回空列表和空metadata
    不回退到其他策略，让上层决定如何处理
    """
    logger = logging.getLogger(__name__)
    repo_root = Path(repo_path)
    all_blocks: List[Block] = []
    all_function_metadata: Dict[int, Dict[str, Optional[str]]] = {}

    # 获取所有支持的文件（不仅是 Python）
    all_files = list(iter_files(repo_root))
    total_files = len(all_files)

    if total_files == 0:
        logger.debug(f"No supported files found in {repo_path}")
        return all_blocks, all_function_metadata

    logger.info(f"Processing {total_files} files in {repo_path}")

    stats = {
        'total_files': total_files,
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'total_blocks': 0,
        'function_blocks': 0,
        'fallback_blocks': 0,
        'files_with_blocks': 0,
    }

    for p in all_files:
        stats['processed'] += 1
        if stats['processed'] % 100 == 0:
            logger.debug(f"Processed {stats['processed']}/{total_files} files in {repo_path} (skipped: {stats['skipped']}, errors: {stats['errors']})")

        # 检查文件大小，跳过过大的文件
        try:
            file_size = p.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB (提高阈值)
                logger.debug(f"Skipping very large file {p.name} ({file_size / 1024 / 1024:.2f}MB)")
                stats['skipped'] += 1
                continue
        except Exception as e:
            logger.warning(f"Error checking file size for {p}: {e}")
            stats['skipped'] += 1
            continue

        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"Error reading {p}: {e}")
                stats['skipped'] += 1
                continue
        except Exception as e:
            logger.warning(f"Error reading {p}: {e}")
            stats['skipped'] += 1
            continue

        rel = str(p.relative_to(repo_root))

        # 使用带 fallback 的函数级切块
        try:
            file_blocks, file_metadata = blocks_function_level_with_fallback(text, rel, block_size)

            # 调整metadata的索引（因为要合并到all_blocks中）
            current_start_idx = len(all_blocks)
            for local_idx, metadata in file_metadata.items():
                global_idx = current_start_idx + local_idx
                all_function_metadata[global_idx] = metadata

            all_blocks.extend(file_blocks)

            # 更新统计
            if file_blocks:
                stats['files_with_blocks'] += 1
                stats['total_blocks'] += len(file_blocks)
                # 分别统计函数块和 fallback 块
                for b in file_blocks:
                    if b.block_type == "function_level":
                        stats['function_blocks'] += 1
                    else:
                        stats['fallback_blocks'] += 1
        except Exception as e:
            logger.error(f"Error processing file {rel}: {e}")
            stats['errors'] += 1
            continue

    logger.info(
        f"Collected {len(all_blocks)} blocks from {repo_path} "
        f"(function: {stats['function_blocks']}, fallback: {stats['fallback_blocks']}, "
        f"processed: {stats['processed']}, skipped: {stats['skipped']}, "
        f"errors: {stats['errors']}, files with blocks: {stats['files_with_blocks']})"
    )

    # 如果没有提取到任何blocks，返回空（让上层决定是否跳过或标记为失败）
    if not all_blocks:
        logger.warning(f"No blocks extracted from {repo_path}")
        return all_blocks, all_function_metadata

    return all_blocks, all_function_metadata


# ============================================================================
# BlockDataset 和编码逻辑
# ============================================================================


class BlockDataset(Dataset):
    def __init__(self, blocks: List[Block], tokenizer, max_length: int, ir_context_tokens: int = 256):
        self.blocks = blocks
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ir_context_tokens = ir_context_tokens

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx: int):
        b = self.blocks[idx]
        is_ir_function = b.block_type == "ir_function" and bool(b.function_text or b.context_text)
        if is_ir_function:
            input_ids, attention_mask, orig_len, truncated = self._encode_ir_function(b)
        else:
            input_ids, attention_mask, orig_len, truncated = self._encode_default(b)
        return input_ids, attention_mask, orig_len, truncated, is_ir_function

    def _pad_to_max_length(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids.size(0) < self.max_length:
            pad_len = self.max_length - input_ids.size(0)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            pad_ids = torch.full((pad_len,), pad_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros((pad_len,), dtype=attention_mask.dtype)
            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=0)
        elif input_ids.size(0) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        return input_ids, attention_mask

    def _encode_default(self, b: Block) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        text = f"file path: {b.file_path}\nlines: {b.start}-{b.end}\n\n{b.content}"
        enc = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        orig_len = int(input_ids.size(0))
        truncated = orig_len > self.max_length

        if truncated:
            input_ids = input_ids[:self.max_length]
        attention_mask = torch.ones_like(input_ids)
        input_ids, attention_mask = self._pad_to_max_length(input_ids, attention_mask)
        return input_ids, attention_mask, orig_len, truncated

    def _encode_ir_function(self, b: Block) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        context_text = b.context_text or ""
        function_text = b.function_text or b.content

        context_ids = []
        if context_text:
            context_ids = self.tokenizer(
                context_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.ir_context_tokens,
            )["input_ids"]

        function_ids = self.tokenizer(
            function_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        orig_len = len(context_ids) + len(function_ids) + special_tokens

        max_function_len = max(self.max_length - special_tokens, 0)
        if len(function_ids) > max_function_len:
            function_ids = function_ids[:max_function_len]

        remaining_context = max(self.max_length - special_tokens - len(function_ids), 0)
        context_ids = context_ids[:remaining_context]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(context_ids + function_ids)
        attention_mask = [1] * len(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        input_ids, attention_mask = self._pad_to_max_length(input_ids, attention_mask)

        truncated = orig_len > self.max_length
        return input_ids, attention_mask, orig_len, truncated


# ============================================================================
# 外部库依赖的切块策略
# ============================================================================


def collect_blocks(repo_path: str, strategy: str, block_size: int, window_size: int, slice_size: int) -> List[Block]:
    logger = logging.getLogger(__name__)
    repo_root = Path(repo_path)
    blocks: List[Block] = []
    for p in iter_files(repo_root):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"Error reading {p}: {e}")
                continue
        except Exception as e:
            logger.warning(f"Error reading {p}: {e}")
            continue
        rel = str(p.relative_to(repo_root))
        if strategy == "fixed":
            blocks.extend(blocks_fixed_lines(text, rel, block_size))
        elif strategy == "sliding":
            blocks.extend(blocks_sliding(text, rel, window_size, slice_size))
        elif strategy == "rl_fixed":
            blocks.extend(blocks_rl_fixed(text, rel))
        elif strategy == "rl_mini":
            blocks.extend(blocks_rl_mini(text, rel))
        elif strategy == "function_level":
            # 使用带 fallback 的函数级切块，确保完整覆盖
            file_blocks, _ = blocks_function_level_with_fallback(text, rel, block_size)
            blocks.extend(file_blocks)
        elif strategy == "ir_function":
            # 论文 IR-based 方法：只索引函数，上下文冗余附加
            file_blocks, _ = blocks_ir_function(text, rel)
            blocks.extend(file_blocks)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    return blocks


def build_context_enhanced_content(
    file_path: str,
    original_content: str,
    start_line: int = 0,
) -> str:
    """
    为代码块添加上下文信息：
    - 文件路径
    - 尝试性的类名（简易启发式）
    - 行号范围
    - 原始代码
    """
    context_parts = [f"File: {file_path}"]

    # 尝试从前若干行中提取类名（仅作辅助上下文，不做强依赖）
    try:
        import re

        header = "\n".join(original_content.splitlines()[:50])
        class_match = re.search(r"class\s+(\w+)", header)
        if class_match:
            context_parts.append(f"Class: {class_match.group(1)}")
    except Exception:
        # 不因上下文增强失败而中断
        pass

    if start_line > 0:
        context_parts.append(f"Lines: {start_line}-")

    context_parts.append("")  # 空行分隔
    context_parts.append("Code:")
    context_parts.append(original_content)

    return "\n".join(context_parts)


def collect_epic_blocks(repo_path: str, config: Optional[EpicSplitterConfig] = None) -> List[Block]:
    """
    使用与 BM25 一致的 EpicSplitter 构建块，并做上下文增强。
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use epic strategy.")
        return []

    # 延迟导入 EpicSplitter，避免不必要的依赖
    try:
        from repo_index.index.epic_split import EpicSplitter
    except ImportError as e:
        logger.error(f"Failed to import EpicSplitter: {e}. Please install required dependencies.")
        return []

    if config is None:
        config = EpicSplitterConfig.from_bm25_config()

    logger.info(f"Using EpicSplitter config: {config}")

    # 1. 读取仓库为文档
    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            exclude=[
                "**/test/**",
                "**/tests/**",
                "**/test_*.py",
                "**/*_test.py",
            ],
            filename_as_id=True,
            required_exts=[".py"],
            recursive=True,
        )
        docs = reader.load_data()
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    # 2. 使用 EpicSplitter 切块
    try:
        splitter = EpicSplitter(
            min_chunk_size=config.min_chunk_size,
            chunk_size=config.chunk_size,
            max_chunk_size=config.max_chunk_size,
            hard_token_limit=config.hard_token_limit,
            max_chunks=config.max_chunks,
            repo_path=repo_path,
        )
        nodes = splitter.get_nodes_from_documents(docs, show_progress=False)
    except Exception as e:
        logger.error(f"EpicSplitter failed for {repo_path}: {e}")
        # Fallback：退回到 rl_mini 行块切分
        logger.warning("Falling back to rl_mini chunking")
        return collect_blocks(repo_path, "rl_mini", block_size=15, window_size=20, slice_size=2)

    blocks: List[Block] = []
    for i, node in enumerate(nodes):
        try:
            meta = getattr(node, "metadata", None) or {}
            file_path = meta.get("file_path", "")
            if not file_path:
                logger.warning(f"Node {i} missing file_path, skipping")
                continue

            start_line = meta.get("start_line")
            end_line = meta.get("end_line")
            content_text = node.get_content()

            # 行号缺失时的推断
            if start_line is None or end_line is None:
                lines = content_text.splitlines()
                start_line = 0
                end_line = len(lines)
                logger.debug(
                    f"Block {i} missing line numbers, inferred {start_line}-{end_line}"
                )

            if start_line < 0 or end_line < start_line:
                logger.warning(
                    f"Invalid line numbers for block {i}: {start_line}-{end_line}, skipping"
                )
                continue

            content = build_context_enhanced_content(
                file_path=file_path,
                original_content=content_text,
                start_line=start_line,
            )

            blocks.append(
                Block(
                    file_path=file_path,
                    start=start_line,
                    end=end_line,
                    content=content,
                    block_type="epic",
                )
            )
        except Exception as e:
            logger.error(f"Error processing node {i} in {repo_path}: {e}")
            continue

    if not blocks:
        raise ValueError(f"No valid blocks extracted from {repo_path}")

    logger.info(f"Collected {len(blocks)} epic blocks from {repo_path}")
    return blocks


def _calculate_chunk_line_numbers(
    original_content: str,
    chunk_content: str,
    previous_chunk_end: int = 0
) -> Tuple[int, int]:
    """
    计算 chunk 在原文件中的实际行号

    Args:
        original_content: 原始文件内容
        chunk_content: chunk 内容
        previous_chunk_end: 上一个 chunk 的结束字符位置（用于优化搜索）

    Returns:
        (start_line, end_line) 元组，0-based 行号
    """
    # 策略1: 精确匹配（处理重叠时可能失败）
    chunk_start_pos = original_content.find(chunk_content, previous_chunk_end)

    if chunk_start_pos >= 0:
        # 找到精确匹配，计算行号
        start_line = original_content[:chunk_start_pos].count('\n')
        end_line = start_line + len(chunk_content.splitlines()) - 1
        return start_line, end_line

    # 策略2: 模糊匹配（使用 chunk 的第一行和最后一行）
    chunk_lines = chunk_content.splitlines()
    if not chunk_lines:
        return 0, 0

    first_line = chunk_lines[0].strip()
    last_line = chunk_lines[-1].strip()

    if not first_line and not last_line:
        # 空 chunk，返回估算值
        return previous_chunk_end // 80, previous_chunk_end // 80  # 假设每行80字符

    # 查找第一行在原文件中的位置
    original_lines = original_content.splitlines()
    start_line = None
    end_line = None

    for i, line in enumerate(original_lines):
        if first_line and first_line in line:
            start_line = i
            break

    # 从后往前查找最后一行
    if last_line:
        for i in range(len(original_lines) - 1, -1, -1):
            if last_line in original_lines[i]:
                end_line = i
                break

    # 如果找到了，返回结果；否则使用估算
    if start_line is not None and end_line is not None:
        return start_line, end_line
    elif start_line is not None:
        return start_line, start_line + len(chunk_lines) - 1
    elif end_line is not None:
        return max(0, end_line - len(chunk_lines) + 1), end_line
    else:
        # 完全找不到，使用启发式估算
        estimated_start = previous_chunk_end // 80  # 假设每行80字符
        return estimated_start, estimated_start + len(chunk_lines) - 1


def _get_or_calculate_line_numbers(
    node,
    original_content: str,
    previous_chunk_end: int = 0
) -> Tuple[int, int]:
    """
    从 node metadata 获取行号，如果不存在则计算

    Args:
        node: LlamaIndex node 对象
        original_content: 原始文件内容
        previous_chunk_end: 上一个 chunk 的结束字符位置（用于优化搜索）

    Returns:
        (start_line, end_line) 元组，0-based 行号
    """
    # 检查 metadata 中是否已有行号
    metadata_start = node.metadata.get('start_line')
    metadata_end = node.metadata.get('end_line')

    # 如果 metadata 中已有有效的行号，直接使用
    if metadata_start is not None and metadata_start >= 0:
        if metadata_end is not None and metadata_end >= metadata_start:
            return metadata_start, metadata_end
        else:
            # 只有 start_line，计算 end_line
            chunk_content = node.get_content()
            return metadata_start, metadata_start + len(chunk_content.splitlines()) - 1

    # metadata 中没有行号，使用辅助函数计算
    chunk_content = node.get_content()
    return _calculate_chunk_line_numbers(
        original_content, chunk_content, previous_chunk_end
    )


def collect_langchain_recursive_blocks(
    repo_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Block]:
    """
    使用 LangChain RecursiveCharacterTextSplitter 收集代码块（简化版）

    Args:
        repo_path: 仓库路径
        chunk_size: 目标块大小（字符数）
        chunk_overlap: 块之间的重叠字符数

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use langchain strategies.")
        return []
    if not HAS_LANGCHAIN:
        logger.error("LangChain text splitters are not available; install langchain-text-splitters to use langchain strategies.")
        return []

    # 1. 读取仓库为文档
    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            exclude=[
                "**/test/**",
                "**/tests/**",
                "**/test_*.py",
                "**/*_test.py",
            ],
            filename_as_id=True,
            required_exts=[".py"],
            recursive=True,
        )
        docs = reader.load_data()
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    # 2. 使用 LangChain RecursiveCharacterTextSplitter 切块
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks = []
        for doc in docs:
            original_content = doc.get_content()  # 保存原始内容
            chunks = splitter.split_text(original_content)

            previous_chunk_end = 0  # 跟踪上一个 chunk 的结束位置
            for chunk in chunks:
                # 计算实际行号
                start_line, end_line = _calculate_chunk_line_numbers(
                    original_content, chunk, previous_chunk_end
                )
                # 更新 previous_chunk_end（估算）
                chunk_pos = original_content.find(chunk, previous_chunk_end)
                if chunk_pos >= 0:
                    previous_chunk_end = chunk_pos + len(chunk)
                else:
                    # 如果找不到，使用行号估算字符位置
                    previous_chunk_end = end_line * 80  # 假设每行80字符

                all_chunks.append({
                    'file_path': doc.metadata.get('file_path', ''),
                    'content': chunk,
                    'start_line': start_line,
                    'end_line': end_line,
                })
    except Exception as e:
        logger.error(f"LangChain RecursiveCharacterTextSplitter failed for {repo_path}: {e}")
        return []

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    for chunk_info in all_chunks:
        file_path = chunk_info['file_path']
        content_text = chunk_info['content']
        start_line = chunk_info['start_line']  # 使用计算出的行号
        end_line = chunk_info['end_line']       # 使用计算出的行号

        if not file_path:
            continue

        content = build_context_enhanced_content(
            file_path=file_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=file_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="langchain_recursive",
            )
        )

    logger.info(f"Collected {len(blocks)} blocks using LangChain RecursiveCharacterTextSplitter from {repo_path}")
    return blocks


def collect_langchain_fixed_blocks(
    repo_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Block]:
    """
    使用 LangChain CharacterTextSplitter 收集代码块（固定字符数分割）

    Args:
        repo_path: 仓库路径
        chunk_size: 目标块大小（字符数）
        chunk_overlap: 块之间的重叠字符数

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use langchain strategies.")
        return []
    if not HAS_LANGCHAIN:
        logger.error("LangChain text splitters are not available; install langchain-text-splitters to use langchain strategies.")
        return []

    # 1. 读取仓库为文档
    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            exclude=[
                "**/test/**",
                "**/tests/**",
                "**/test_*.py",
                "**/*_test.py",
            ],
            filename_as_id=True,
            required_exts=[".py"],
            recursive=True,
        )
        docs = reader.load_data()
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    # 2. 使用 LangChain CharacterTextSplitter 切块
    try:
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks = []
        for doc in docs:
            original_content = doc.get_content()  # 保存原始内容
            chunks = splitter.split_text(original_content)

            previous_chunk_end = 0  # 跟踪上一个 chunk 的结束位置
            for chunk in chunks:
                # 计算实际行号
                start_line, end_line = _calculate_chunk_line_numbers(
                    original_content, chunk, previous_chunk_end
                )
                # 更新 previous_chunk_end（估算）
                chunk_pos = original_content.find(chunk, previous_chunk_end)
                if chunk_pos >= 0:
                    previous_chunk_end = chunk_pos + len(chunk)
                else:
                    # 如果找不到，使用行号估算字符位置
                    previous_chunk_end = end_line * 80  # 假设每行80字符

                all_chunks.append({
                    'file_path': doc.metadata.get('file_path', ''),
                    'content': chunk,
                    'start_line': start_line,
                    'end_line': end_line,
                })
    except Exception as e:
        logger.error(f"LangChain CharacterTextSplitter failed for {repo_path}: {e}")
        return []

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    for chunk_info in all_chunks:
        file_path = chunk_info['file_path']
        content_text = chunk_info['content']
        start_line = chunk_info['start_line']  # 使用计算出的行号
        end_line = chunk_info['end_line']       # 使用计算出的行号

        if not file_path:
            continue

        content = build_context_enhanced_content(
            file_path=file_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=file_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="langchain_fixed",
            )
        )

    logger.info(f"Collected {len(blocks)} blocks using LangChain CharacterTextSplitter from {repo_path}")
    return blocks


def collect_langchain_token_blocks(
    repo_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Block]:
    """
    使用 LangChain TokenTextSplitter 收集代码块（按 token 数分割）

    Args:
        repo_path: 仓库路径
        chunk_size: 目标块大小（token数）
        chunk_overlap: 块之间的重叠token数

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use langchain strategies.")
        return []
    if not HAS_LANGCHAIN:
        logger.error("LangChain text splitters are not available; install langchain-text-splitters to use langchain strategies.")
        return []

    # 1. 读取仓库为文档
    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            exclude=[
                "**/test/**",
                "**/tests/**",
                "**/test_*.py",
                "**/*_test.py",
            ],
            filename_as_id=True,
            required_exts=[".py"],
            recursive=True,
        )
        docs = reader.load_data()
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    # 2. 使用 LangChain TokenTextSplitter 切块
    try:
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            disallowed_special=(),  # 允许所有特殊 token，避免遇到 '<|endoftext|>' 等特殊 token 时报错
        )

        all_chunks = []
        for doc in docs:
            original_content = doc.get_content()  # 保存原始内容
            chunks = splitter.split_text(original_content)

            previous_chunk_end = 0  # 跟踪上一个 chunk 的结束位置
            for chunk in chunks:
                # 计算实际行号
                start_line, end_line = _calculate_chunk_line_numbers(
                    original_content, chunk, previous_chunk_end
                )
                # 更新 previous_chunk_end（估算）
                chunk_pos = original_content.find(chunk, previous_chunk_end)
                if chunk_pos >= 0:
                    previous_chunk_end = chunk_pos + len(chunk)
                else:
                    # 如果找不到，使用行号估算字符位置
                    previous_chunk_end = end_line * 80  # 假设每行80字符

                all_chunks.append({
                    'file_path': doc.metadata.get('file_path', ''),
                    'content': chunk,
                    'start_line': start_line,
                    'end_line': end_line,
                })
    except Exception as e:
        logger.error(f"LangChain TokenTextSplitter failed for {repo_path}: {e}")
        return []

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    for chunk_info in all_chunks:
        file_path = chunk_info['file_path']
        content_text = chunk_info['content']
        start_line = chunk_info['start_line']  # 使用计算出的行号
        end_line = chunk_info['end_line']       # 使用计算出的行号

        if not file_path:
            continue

        content = build_context_enhanced_content(
            file_path=file_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=file_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="langchain_token",
            )
        )

    logger.info(f"Collected {len(blocks)} blocks using LangChain TokenTextSplitter from {repo_path}")
    return blocks


def collect_llamaindex_code_blocks(
    repo_path: str,
    language: str = "python",
    chunk_lines: int = 40,
    chunk_lines_overlap: int = 15,
    max_chars: int = 1500,
) -> List[Block]:
    """
    使用 LlamaIndex CodeSplitter 收集代码块（代码专用，基于AST）

    Args:
        repo_path: 仓库路径
        language: 编程语言（默认python）
        chunk_lines: 每个块的代码行数
        chunk_lines_overlap: 块之间的重叠行数
        max_chars: 每个块的最大字符数

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use llamaindex_code strategy.")
        return []

    # 1. 读取仓库为文档（复用现有模式）
    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            exclude=[
                "**/test/**",
                "**/tests/**",
                "**/test_*.py",
                "**/*_test.py",
            ],
            filename_as_id=True,
            required_exts=[".py"],
            recursive=True,
        )
        docs = reader.load_data()
        if not docs:
            print(f'\n[DEBUG] {repo_path}: No documents loaded (after exclusions)')
            logger.warning(f"No documents loaded from {repo_path} (after exclusions)")
            return []
        print(f'\n[DEBUG] {repo_path}: Loaded {len(docs)} documents')
        logger.info(f"Loaded {len(docs)} documents from {repo_path}")
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    # 2. 使用 LlamaIndex CodeSplitter 切块
    try:
        splitter = CodeSplitter(
            language=language,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
        )

        nodes = splitter.get_nodes_from_documents(docs)
        print(f'\n[DEBUG] {repo_path}: CodeSplitter returned {len(nodes) if nodes else 0} nodes')
        if not nodes:
            print(f'\n[DEBUG] {repo_path}: No nodes generated from {len(docs)} documents')
            logger.warning(f"No nodes generated from {len(docs)} documents for {repo_path}")
            return []
        print(f'\n[DEBUG] {repo_path}: Generated {len(nodes)} nodes from {len(docs)} documents')
        logger.info(f"Generated {len(nodes)} nodes from {len(docs)} documents for {repo_path}")
    except Exception as e:
        print(f'\n[DEBUG] {repo_path}: CodeSplitter Exception: {type(e).__name__}: {e}')
        logger.error(f"LlamaIndex CodeSplitter failed for {repo_path}: {e}")
        logger.warning(f"Falling back to fixed chunking strategy for {repo_path}")
        # 降级处理：使用 fixed 分块策略作为备用
        try:
            blocks = collect_blocks(repo_path, "fixed", block_size=40, window_size=50, slice_size=10)
            if blocks:
                print(f'\n[DEBUG] {repo_path}: Fallback to fixed strategy generated {len(blocks)} blocks')
                logger.info(f"Fallback to fixed strategy generated {len(blocks)} blocks for {repo_path}")
                return blocks
        except Exception as fallback_error:
            logger.error(f"Fallback strategy also failed for {repo_path}: {fallback_error}")
        return []

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    # 按文件分组处理，以便跟踪 previous_chunk_end
    file_to_docs = {}
    for doc in docs:
        file_path = doc.metadata.get('file_path', '') or doc.metadata.get('file_name', '')
        if file_path:
            if file_path not in file_to_docs:
                file_to_docs[file_path] = []
            file_to_docs[file_path].append(doc)

    # 为每个文件建立内容映射
    file_contents = {}
    for file_path, file_docs in file_to_docs.items():
        # 合并同一文件的所有文档内容
        file_contents[file_path] = '\n'.join([doc.get_content() for doc in file_docs])
    previous_chunk_end_by_file = {}

    for node in nodes:
        file_path = node.metadata.get('file_path', '') or node.metadata.get('file_name', '')
        if not file_path:
            continue

        content_text = node.get_content()
        if not content_text:
            continue

        # 获取或计算行号
        original_content = file_contents.get(file_path, content_text)
        prev_end = previous_chunk_end_by_file.get(file_path, 0)
        start_line, end_line = _get_or_calculate_line_numbers(
            node, original_content, previous_chunk_end=prev_end
        )
        chunk_pos = original_content.find(content_text, prev_end)
        if chunk_pos >= 0:
            previous_chunk_end_by_file[file_path] = chunk_pos + len(content_text)
        else:
            previous_chunk_end_by_file[file_path] = max(prev_end, end_line * 80)

        content = build_context_enhanced_content(
            file_path=file_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=file_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="llamaindex_code",
            )
        )

    if not blocks:
        print(f'\n[DEBUG] {repo_path}: No blocks created from {len(nodes)} nodes')
        logger.warning(f"No blocks created from {len(nodes)} nodes for {repo_path}")
    else:
        print(f'\n[DEBUG] {repo_path}: Collected {len(blocks)} blocks from {len(nodes)} nodes')
        logger.info(f"Collected {len(blocks)} blocks using LlamaIndex CodeSplitter from {repo_path}")
    return blocks


def collect_llamaindex_sentence_blocks(
    repo_path: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> List[Block]:
    """
    使用 LlamaIndex SentenceSplitter 收集代码块（按句子分割）

    Args:
        repo_path: 仓库路径
        chunk_size: 目标块大小（字符数）
        chunk_overlap: 块之间的重叠字符数

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use llamaindex_sentence strategy.")
        return []

    # 1. 读取仓库为文档
    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            exclude=[
                "**/test/**",
                "**/tests/**",
                "**/test_*.py",
                "**/*_test.py",
            ],
            filename_as_id=True,
            required_exts=[".py"],
            recursive=True,
        )
        docs = reader.load_data()
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    # 2. 使用 LlamaIndex SentenceSplitter 切块
    try:
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        nodes = splitter.get_nodes_from_documents(docs)
    except Exception as e:
        logger.error(f"LlamaIndex SentenceSplitter failed for {repo_path}: {e}")
        return []

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    # 按文件分组处理，以便跟踪 previous_chunk_end
    file_to_docs = {}
    for doc in docs:
        file_path = doc.metadata.get('file_path', '') or doc.metadata.get('file_name', '')
        if file_path:
            if file_path not in file_to_docs:
                file_to_docs[file_path] = []
            file_to_docs[file_path].append(doc)

    # 为每个文件建立内容映射
    file_contents = {}
    for file_path, file_docs in file_to_docs.items():
        # 合并同一文件的所有文档内容
        file_contents[file_path] = '\n'.join([doc.get_content() for doc in file_docs])
    previous_chunk_end_by_file = {}

    for node in nodes:
        file_path = node.metadata.get('file_path', '') or node.metadata.get('file_name', '')
        if not file_path:
            continue

        content_text = node.get_content()
        if not content_text:
            continue

        # 获取或计算行号
        original_content = file_contents.get(file_path, content_text)
        prev_end = previous_chunk_end_by_file.get(file_path, 0)
        start_line, end_line = _get_or_calculate_line_numbers(
            node, original_content, previous_chunk_end=prev_end
        )
        chunk_pos = original_content.find(content_text, prev_end)
        if chunk_pos >= 0:
            previous_chunk_end_by_file[file_path] = chunk_pos + len(content_text)
        else:
            previous_chunk_end_by_file[file_path] = max(prev_end, end_line * 80)

        content = build_context_enhanced_content(
            file_path=file_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=file_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="llamaindex_sentence",
            )
        )

    logger.info(f"Collected {len(blocks)} blocks using LlamaIndex SentenceSplitter from {repo_path}")
    return blocks


def collect_llamaindex_token_blocks(
    repo_path: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 20,
    separator: str = " ",
) -> List[Block]:
    """
    使用 LlamaIndex TokenTextSplitter 收集代码块（按token分割）

    Args:
        repo_path: 仓库路径
        chunk_size: 目标块大小（token数）
        chunk_overlap: 块之间的重叠token数
        separator: 分隔符

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use llamaindex_token strategy.")
        return []

    # 1. 读取仓库为文档
    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            exclude=[
                "**/test/**",
                "**/tests/**",
                "**/test_*.py",
                "**/*_test.py",
            ],
            filename_as_id=True,
            required_exts=[".py"],
            recursive=True,
        )
        docs = reader.load_data()
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    # 2. 使用 LlamaIndex TokenTextSplitter 切块
    try:
        splitter = LlamaIndexTokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
        )

        nodes = splitter.get_nodes_from_documents(docs)
    except Exception as e:
        logger.error(f"LlamaIndex TokenTextSplitter failed for {repo_path}: {e}")
        return []

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    # 按文件分组处理，以便跟踪 previous_chunk_end
    file_to_docs = {}
    for doc in docs:
        file_path = doc.metadata.get('file_path', '') or doc.metadata.get('file_name', '')
        if file_path:
            if file_path not in file_to_docs:
                file_to_docs[file_path] = []
            file_to_docs[file_path].append(doc)

    # 为每个文件建立内容映射
    file_contents = {}
    for file_path, file_docs in file_to_docs.items():
        # 合并同一文件的所有文档内容
        file_contents[file_path] = '\n'.join([doc.get_content() for doc in file_docs])
    previous_chunk_end_by_file = {}

    for node in nodes:
        file_path = node.metadata.get('file_path', '') or node.metadata.get('file_name', '')
        if not file_path:
            continue

        content_text = node.get_content()
        if not content_text:
            continue

        # 获取或计算行号
        original_content = file_contents.get(file_path, content_text)
        prev_end = previous_chunk_end_by_file.get(file_path, 0)
        start_line, end_line = _get_or_calculate_line_numbers(
            node, original_content, previous_chunk_end=prev_end
        )
        chunk_pos = original_content.find(content_text, prev_end)
        if chunk_pos >= 0:
            previous_chunk_end_by_file[file_path] = chunk_pos + len(content_text)
        else:
            previous_chunk_end_by_file[file_path] = max(prev_end, end_line * 80)

        content = build_context_enhanced_content(
            file_path=file_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=file_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="llamaindex_token",
            )
        )

    logger.info(f"Collected {len(blocks)} blocks using LlamaIndex TokenTextSplitter from {repo_path}")
    return blocks


def collect_llamaindex_semantic_blocks(
    repo_path: str,
    buffer_size: int = 3,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Block]:
    """
    使用 LlamaIndex SemanticSplitterNodeParser 收集代码块（基于语义相似性）

    Args:
        repo_path: 仓库路径
        buffer_size: 缓冲区大小（用于语义分割）
        model_name: HuggingFace embedding 模型名称（默认使用轻量级模型）

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use llamaindex_semantic strategy.")
        return []
    if not HAS_LLAMA_INDEX_HF:
        logger.error("HuggingFaceEmbedding is not available; install LlamaIndex HuggingFace embeddings to use llamaindex_semantic strategy.")
        return []

    # 1. 读取仓库为文档
    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            exclude=[
                "**/test/**",
                "**/tests/**",
                "**/test_*.py",
                "**/*_test.py",
            ],
            filename_as_id=True,
            required_exts=[".py"],
            recursive=True,
        )
        docs = reader.load_data()
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    # 2. 使用本地 HuggingFace embedding 模型
    try:
        # 使用本地模型，无需 API key
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            embed_model=embed_model,
        )

        nodes = splitter.get_nodes_from_documents(docs)
    except Exception as e:
        logger.error(f"LlamaIndex SemanticSplitter failed for {repo_path}: {e}")
        return []

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    # 按文件分组处理，以便跟踪 previous_chunk_end
    file_to_docs = {}
    for doc in docs:
        file_path = doc.metadata.get('file_path', '') or doc.metadata.get('file_name', '')
        if file_path:
            if file_path not in file_to_docs:
                file_to_docs[file_path] = []
            file_to_docs[file_path].append(doc)

    # 为每个文件建立内容映射
    file_contents = {}
    for file_path, file_docs in file_to_docs.items():
        # 合并同一文件的所有文档内容
        file_contents[file_path] = '\n'.join([doc.get_content() for doc in file_docs])
    previous_chunk_end_by_file = {}

    for node in nodes:
        file_path = node.metadata.get('file_path', '') or node.metadata.get('file_name', '')
        if not file_path:
            continue

        content_text = node.get_content()
        if not content_text:
            continue

        # 获取或计算行号
        original_content = file_contents.get(file_path, content_text)
        prev_end = previous_chunk_end_by_file.get(file_path, 0)
        start_line, end_line = _get_or_calculate_line_numbers(
            node, original_content, previous_chunk_end=prev_end
        )
        chunk_pos = original_content.find(content_text, prev_end)
        if chunk_pos >= 0:
            previous_chunk_end_by_file[file_path] = chunk_pos + len(content_text)
        else:
            previous_chunk_end_by_file[file_path] = max(prev_end, end_line * 80)

        content = build_context_enhanced_content(
            file_path=file_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=file_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="llamaindex_semantic",
            )
        )

    logger.info(f"Collected {len(blocks)} blocks using LlamaIndex SemanticSplitter from {repo_path}")
    return blocks


# ============================================================================
# 工具函数
# ============================================================================


def list_folders(path: str) -> List[str]:
    return [p.name for p in Path(path).iterdir() if p.is_dir()]


def instance_id_to_repo_name(instance_id: str) -> str:
    """将 instance_id 转换为 repo_name（去掉 issue 编号后缀）"""
    import re
    repo_part = re.sub(r'-\d+$', '', instance_id)
    return repo_part.replace('__', '_')


def resolve_model_name(model_name: str) -> str:
    if os.path.isabs(model_name):
        return model_name
    if os.path.exists(model_name):
        return os.path.abspath(model_name)
    project_candidate = _project_root / model_name
    if project_candidate.exists():
        return str(project_candidate.resolve())
    return model_name


def save_index(
    output_dir: Path,
    embeddings: torch.Tensor,
    blocks: List[Block],
    strategy: str,
    span_ids_map: Optional[Dict[int, List[str]]] = None,
    epic_config: Optional[EpicSplitterConfig] = None,
    function_metadata: Optional[Dict[int, Dict[str, Optional[str]]]] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_dir / "embeddings.pt")
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for idx, b in enumerate(blocks):
            metadata = {
                "block_id": idx,
                "file_path": b.file_path,
                "start_line": b.start,
                "end_line": b.end,
                "block_type": b.block_type,
                "strategy": strategy,
            }
            # 如果有span_ids，添加到metadata
            if span_ids_map and idx in span_ids_map:
                metadata["span_ids"] = span_ids_map[idx]

            # epic 策略的额外元数据，便于追溯与复现
            if strategy == "epic":
                cfg = epic_config or EpicSplitterConfig.from_bm25_config()
                metadata.update(
                    {
                        "epic_splitter_version": "1.0",
                        "chunking_config": {
                            "min_chunk_size": cfg.min_chunk_size,
                            "chunk_size": cfg.chunk_size,
                            "max_chunk_size": cfg.max_chunk_size,
                            "hard_token_limit": cfg.hard_token_limit,
                            "max_chunks": cfg.max_chunks,
                        },
                        "context_enhanced": True,
                        "chunk_tokens": len(b.content.split()) if b.content else 0,
                        "created_at": datetime.now().isoformat(),
                    }
                )

            # function_level 策略的额外元数据
            if strategy == "function_level" and function_metadata and idx in function_metadata:
                metadata.update({
                    "qualified_name": function_metadata[idx]["qualified_name"],
                    "class_name": function_metadata[idx]["class_name"],
                    "function_name": function_metadata[idx]["function_name"],
                    "function_level_version": "1.0",
                    "created_at": datetime.now().isoformat(),
                })

            # ir_function 策略的额外元数据
            if strategy == "ir_function" and function_metadata and idx in function_metadata:
                metadata.update({
                    "qualified_name": function_metadata[idx]["qualified_name"],
                    "class_name": function_metadata[idx]["class_name"],
                    "function_name": function_metadata[idx]["function_name"],
                    "ir_function_version": "1.0",
                    "created_at": datetime.now().isoformat(),
                })

            f.write(json.dumps(metadata) + "\n")


# ============================================================================
# 外部库依赖检查
# ============================================================================


try:
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import (
        CodeSplitter,
        SentenceSplitter,
        TokenTextSplitter as LlamaIndexTokenTextSplitter,
        SemanticSplitterNodeParser,
    )
    HAS_LLAMA_INDEX = True
except ImportError:
    HAS_LLAMA_INDEX = False
    SimpleDirectoryReader = None
    CodeSplitter = None
    SentenceSplitter = None
    LlamaIndexTokenTextSplitter = None
    SemanticSplitterNodeParser = None

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    HAS_LLAMA_INDEX_HF = True
except ImportError:
    HAS_LLAMA_INDEX_HF = False
    HuggingFaceEmbedding = None

try:
    from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
        TokenTextSplitter,
    )
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    CharacterTextSplitter = None
    RecursiveCharacterTextSplitter = None
    TokenTextSplitter = None