#!/bin/bash
# 删除索引并重新构建

cd /workspace/locbench/IR-based

echo "删除现有索引目录..."
rm -rf new_index_data/llamacode_CodeRankEmbed/

echo ""
echo "开始重新构建索引（所有165个仓库）..."
echo ""

python method/index/batch_build_index.py \
    --repo_path /workspace/locbench/repos/locbench_repos \
    --strategy llamaindex_code \
    --index_dir new_index_data/llamacode_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 4 \
    --gpu_ids 0,1,2,3 \
    --batch_size 32

echo ""
echo "验证索引构建结果..."
python3 -c "
from pathlib import Path
index_base = Path('new_index_data/llamacode_CodeRankEmbed/dense_index_llamaindex_code')
if index_base.exists():
    repos = [d.name for d in index_base.iterdir() if d.is_dir()]
    print(f'索引目录中的仓库数: {len(repos)}')
    
    # 检查缺失的仓库
    missing_repos = ['ckan_ckan', 'pylint-dev_pylint']
    for repo_name in missing_repos:
        repo_dir = index_base / repo_name
        embeddings_file = repo_dir / 'embeddings.pt'
        if embeddings_file.exists():
            print(f'✓ {repo_name}: 索引已创建')
        else:
            print(f'✗ {repo_name}: 索引仍然缺失')
else:
    print('索引目录不存在')
"
