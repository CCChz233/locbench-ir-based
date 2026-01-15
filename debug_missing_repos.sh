#!/bin/bash
# 调试缺失的仓库，查看详细的日志输出

cd /workspace/locbench/IR-based

echo "开始调试构建，查看详细日志..."
echo ""

# 只处理这两个缺失的仓库（通过限制仓库列表）
python method/index/batch_build_index.py \
    --repo_path /workspace/locbench/repos/locbench_repos \
    --strategy llamaindex_code \
    --index_dir new_index_data/llamacode_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 1 \
    --gpu_ids 1 \
    --batch_size 32 2>&1 | grep -E "(ckan_ckan|pylint-dev_pylint|Warning|Error|Loaded|Generated|Collected|blocks)" | head -50
