#!/usr/bin/env python
"""
为缺失的仓库构建索引
"""
import sys
from pathlib import Path

# 添加路径
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# LocAgent 依赖已复制到 IR-based 下，不再需要添加路径

import subprocess
import os

# 缺失的仓库列表
missing_repos = ['ckan_ckan', 'pylint-dev_pylint']

# 索引构建参数
repo_path = '/workspace/locbench/repos/locbench_repos'
strategy = 'llamaindex_code'
index_dir = 'new_index_data/llamacode_CodeRankEmbed'
model_name = 'models/CodeRankEmbed'
gpu_ids = '1'  # 使用单个GPU
batch_size = 32

print(f"开始为 {len(missing_repos)} 个缺失的仓库构建索引...")
print(f"缺失的仓库: {missing_repos}")

# 构建命令
cmd = [
    'python', 'method/index/batch_build_index.py',
    '--repo_path', repo_path,
    '--strategy', strategy,
    '--index_dir', index_dir,
    '--model_name', model_name,
    '--trust_remote_code',
    '--num_processes', '1',
    '--gpu_ids', gpu_ids,
    '--batch_size', str(batch_size),
]

print(f"\n执行命令: {' '.join(cmd)}")
print(f"\n注意: 由于脚本会处理所有仓库，但已存在的索引会被跳过，")
print(f"所以这个命令只会为缺失的仓库构建索引。\n")

# 但是，由于batch_build_index.py会处理所有仓库，我们需要另一种方法
# 我们可以创建一个临时目录，只包含这两个仓库的符号链接
# 或者直接运行，因为已存在的索引会被跳过

# 更简单的方法：直接运行，已存在的索引会被跳过（第2278-2288行的逻辑）
os.chdir(_project_root)
result = subprocess.run(cmd, check=False)

if result.returncode == 0:
    print("\n✓ 索引构建完成！")
    
    # 验证索引是否创建
    index_base = Path(index_dir) / f'dense_index_{strategy}'
    for repo_name in missing_repos:
        repo_index_dir = index_base / repo_name
        embeddings_file = repo_index_dir / 'embeddings.pt'
        metadata_file = repo_index_dir / 'metadata.jsonl'
        
        if embeddings_file.exists() and metadata_file.exists():
            print(f"  ✓ {repo_name}: 索引已创建")
        else:
            print(f"  ✗ {repo_name}: 索引仍然缺失")
else:
    print(f"\n✗ 索引构建失败，退出码: {result.returncode}")
    sys.exit(1)
