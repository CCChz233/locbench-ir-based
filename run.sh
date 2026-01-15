#ir_base和llamacode的模型都使用CodeRankEmbed_768

python method/index/batch_build_index.py \
    --repo_path /workspace/locbench/repos/locbench_repos \
    --strategy ir_function \
    --index_dir new_index_data/ir_function_CodeRankEmbed_768 \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 4 \
    --gpu_ids 4,5,6,7 \
    --batch_size 32


python method/index/batch_build_index.py \
    --repo_path /workspace/locbench/repos/locbench_repos \
    --strategy llamaindex_code \
    --index_dir new_index_data/llamacode_CodeRankEmbed_768 \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 4 \
    --gpu_ids 4,5,6,7 \
    --batch_size 32

python method/dense/run_with_index.py \
  --index_dir new_index_data/llamacode_CodeRankEmbed_768/dense_index_llamacode \
  --dataset_path /workspace/locbench/Loc-Bench_V1_dataset.jsonl \
  --output_folder new_outputs/llamacode_CodeRankEmbed_768 \
  --model_name models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --repos_root /workspace/locbench/repos/locbench_repos \
  --gpu_id 1 \
  --batch_size 16

python method/dense/run_with_index.py \
  --index_dir new_index_data/ir_function_CodeRankEmbed_768/dense_index_ir_function \
  --dataset_path /workspace/locbench/Loc-Bench_V1_dataset.jsonl \
  --output_folder new_outputs/ir_function_CodeRankEmbed_768 \
  --model_name models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --repos_root /workspace/locbench/repos/locbench_repos \
  --gpu_id 1 \
  --batch_size 16