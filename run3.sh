# CodeRankEmbed - 使用 batch_build_index.py（不需要 --force_transformers_backend）
python method/index/batch_build_index.py \
    --repo_path /workspace/locbench/repos/locbench_repos \
    --strategy ir_function \
    --index_dir /workspace/locbench/IR-based/new_index_data/ir_function_CodeRankEmbed_4096 \
    --model_name /workspace/locbench/IR-based/models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 3 \
    --gpu_ids 0,1,2 \
    --batch_size 8

python method/index/batch_build_index.py \
    --repo_path /workspace/locbench/repos/locbench_repos \
    --strategy llamaindex_code \
    --index_dir /workspace/locbench/IR-based/new_index_data/llamaindex_code_CodeRankEmbed_4096 \
    --model_name /workspace/locbench/IR-based/models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 4 \
    --gpu_ids 0,1,2,3 \
    --batch_size 8

# SFR - 使用 batch_build_index_sfr.py（不需要 --force_sentence_transformer）
python method/index/batch_build_index_sfr.py \
    --repo_path /workspace/locbench/repos/locbench_repos \
    --strategy llamaindex_code \
    --index_dir /workspace/locbench/IR-based/new_index_data/llamaindex_code_SFR-Embedding-Code-2B_R \
    --model_name /workspace/locbench/IR-based/models/hf_bundle/Salesforce__SFR-Embedding-Code-2B_R \
    --trust_remote_code \
    --num_processes 3 \
    --gpu_ids 0,1,2 \
    --batch_size 16