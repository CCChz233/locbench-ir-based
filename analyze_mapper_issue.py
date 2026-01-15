#!/usr/bin/env python
"""
分析为什么mapper返回空的found_modules和found_entities
"""

import json
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, '/workspace/locbench/IR-based')
# LocAgent 依赖已复制到 IR-based 下，不再需要添加路径

from method.mapping.ast_based.mapper import ASTBasedMapper
from method.utils import instance_id_to_repo_name

def clean_file_path(file_path: str, repo_name: str) -> str:
    """清理文件路径，使其与GT格式一致"""
    # 移除repo_name前缀（如果存在）
    if file_path.startswith(repo_name + '/'):
        file_path = file_path[len(repo_name) + 1:]
    elif file_path.startswith(repo_name + '_'):
        # 处理下划线分隔的情况
        file_path = file_path[len(repo_name) + 1:]
    return file_path

# 读取一个输出实例
output_file = Path("new_outputs/ir_function_CodeRankEmbed/loc_outputs.jsonl")
with open(output_file, 'r') as f:
    output_data = [json.loads(line) for line in f]

# 找一个有found_files的实例
test_instance = None
for d in output_data:
    if d.get('found_files'):
        test_instance = d
        break

if not test_instance:
    print("没有找到有found_files的实例")
    sys.exit(1)

print(f"测试实例: {test_instance['instance_id']}")
print(f"找到的文件数: {len(test_instance['found_files'])}")
print(f"找到的模块数: {len(test_instance.get('found_modules', []))}")
print(f"找到的实体数: {len(test_instance.get('found_entities', []))}")

# 加载对应的索引metadata
repo_name = instance_id_to_repo_name(test_instance['instance_id'])
index_dir = Path("new_index_data/ir_function_CodeRankEmbed/dense_index_ir_function")
metadata_file = index_dir / repo_name / "metadata.jsonl"

if not metadata_file.exists():
    print(f"索引文件不存在: {metadata_file}")
    sys.exit(1)

with open(metadata_file, 'r') as f:
    metadata = [json.loads(line) for line in f]

print(f"\n索引metadata总数: {len(metadata)}")
print(f"\n前5个代码块的metadata:")
for i, m in enumerate(metadata[:5]):
    print(f"  代码块 {i}: file_path={m.get('file_path')}, start_line={m.get('start_line')}, end_line={m.get('end_line')}")

# 模拟检索过程：取前10个代码块
top_blocks = []
for idx in range(min(10, len(metadata))):
    block = metadata[idx].copy()
    original_path = block.get('file_path', '')
    if original_path:
        block['file_path'] = clean_file_path(original_path, repo_name)
    top_blocks.append(block)

print(f"\n清理后的top_blocks (前3个):")
for i, block in enumerate(top_blocks[:3]):
    print(f"  代码块 {i}: file_path={block.get('file_path')}, start_line={block.get('start_line')}, end_line={block.get('end_line')}")

# 测试mapper
mapper = ASTBasedMapper(repos_root="/workspace/locbench/repos/locbench_repos")

print(f"\n测试mapper.map_blocks_to_entities...")
found_modules, found_entities = mapper.map_blocks_to_entities(
    blocks=top_blocks,
    instance_id=test_instance['instance_id'],
    top_k_modules=15,
    top_k_entities=50,
)

print(f"结果:")
print(f"  找到的模块数: {len(found_modules)}")
print(f"  找到的实体数: {len(found_entities)}")
if found_modules:
    print(f"  模块示例: {found_modules[:3]}")
if found_entities:
    print(f"  实体示例: {found_entities[:3]}")

# 检查文件是否存在
print(f"\n检查文件是否存在:")
repo_root = Path("/workspace/locbench/repos/locbench_repos") / repo_name
for block in top_blocks[:3]:
    file_path = block.get('file_path')
    full_path = repo_root / file_path
    exists = full_path.exists()
    print(f"  {file_path} -> {full_path} (存在: {exists})")
    
    if exists:
        # 尝试构建映射
        from method.mapping.ast_based.mapper import build_line_to_entity_map
        line_map = build_line_to_entity_map(file_path, str(repo_root))
        print(f"    映射表大小: {len(line_map)}")
        if line_map:
            print(f"    示例映射: {list(line_map.items())[:3]}")
