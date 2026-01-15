#
cd "$(dirname "$0")" || exit 1

python method/index/batch_build_index.py \
    --repo_path /workspace/locbench/repos/locbench_repos \
    --strategy llamaindex_code \
    --index_dir new_index_data/llamacode_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 4 \
    --gpu_ids 0,1,2,3 \
    --batch_size 32

python method/dense/run_with_index.py \
  --index_dir new_index_data/llamacode_CodeRankEmbed/dense_index_llamaindex_code \
  --dataset_path /workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --output_folder new_outputs/llamacode_CodeRankEmbed \
  --model_name models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --repos_root /workspace/locbench/repos/locbench_repos \
  --gpu_id 1 \
  --batch_size 16