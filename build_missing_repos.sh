#!/bin/bash
# 为缺失的2个仓库构建索引
# 由于batch_build_index.py会跳过已存在的索引，重新运行会只为缺失的仓库构建索引

cd /workspace/locbench/IR-based

echo "开始为缺失的2个仓库构建索引: ckan_ckan, pylint-dev_pylint"
echo "已存在的索引会被跳过，只处理缺失的仓库"
echo ""

python method/index/batch_build_index.py \
    --repo_path /workspace/locbench/repos/locbench_repos \
    --strategy llamaindex_code \
    --index_dir new_index_data/llamacode_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 1 \
    --gpu_ids 1 \
    --batch_size 32

echo ""
echo "验证索引是否已创建..."
python3 -c "
from pathlib import Path
missing_repos = ['ckan_ckan', 'pylint-dev_pylint']
index_base = Path('new_index_data/llamacode_CodeRankEmbed/dense_index_llamaindex_code')
for repo_name in missing_repos:
    repo_index_dir = index_base / repo_name
    embeddings_file = repo_index_dir / 'embeddings.pt'
    metadata_file = repo_index_dir / 'metadata.jsonl'
    if embeddings_file.exists() and metadata_file.exists():
        print(f'✓ {repo_name}: 索引已创建')
    else:
        print(f'✗ {repo_name}: 索引仍然缺失')
"
