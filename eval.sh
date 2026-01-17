python method/dense/run_with_index.py \
  --index_dir /workspace/locbench/IR-based/new_index_data/llamacode_CodeRankEmbed_768/dense_index_llamaindex_code \
  --dataset_path /workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --output_folder new_outputs/llamacode_CodeRankEmbed_768 \
  --model_name models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --repos_root /workspace/locbench/repos/locbench_repos \
  --gpu_id 1 \
  --batch_size 16

python method/dense/run_with_index.py \
  --index_dir new_index_data/ir_function_CodeRankEmbed_768/dense_index_ir_function \
  --dataset_path /workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --output_folder new_outputs/ir_function_CodeRankEmbed_768 \
  --model_name models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --repos_root /workspace/locbench/repos/locbench_repos \
  --gpu_id 1 \
  --batch_size 16