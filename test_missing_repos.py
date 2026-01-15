#!/usr/bin/env python
"""
测试缺失的两个仓库是否能生成blocks
"""
import sys
from pathlib import Path

# 添加路径
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# LocAgent 依赖已复制到 IR-based 下，不再需要添加路径

# 直接从当前目录导入（IR-based目录下的文件）
from method.index.batch_build_index import collect_llamaindex_code_blocks

missing_repos = ['ckan_ckan', 'pylint-dev_pylint']
repo_path = '/workspace/locbench/repos/locbench_repos'

for repo_name in missing_repos:
    repo_dir = Path(repo_path) / repo_name
    print(f'\n{"="*60}')
    print(f'测试仓库: {repo_name}')
    print(f'仓库目录: {repo_dir}')
    print(f'目录存在: {repo_dir.exists()}')
    
    if not repo_dir.exists():
        print(f'  ✗ 仓库目录不存在！')
        continue
    
    try:
        blocks = collect_llamaindex_code_blocks(
            str(repo_dir),
            language='python',
            chunk_lines=40,
            chunk_lines_overlap=15,
            max_chars=1500,
        )
        print(f'  ✓ 成功生成 {len(blocks)} 个blocks')
        if len(blocks) == 0:
            print(f'  ⚠️  警告: blocks为空！这会导致被跳过')
        else:
            print(f'  ✓ blocks不为空，可以构建索引')
    except Exception as e:
        print(f'  ✗ 错误: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
