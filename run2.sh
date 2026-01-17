# CodeRankEmbed 评估 - 使用 run_with_index.py（不需要额外参数）
python method/dense/run_with_index.py \
  --index_dir /workspace/locbench/IR-based/new_index_data/ir_function_CodeRankEmbed2048/dense_index_ir_function \
  --dataset_path /workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --output_folder /workspace/locbench/IR-based/new_outputs \
  --model_name /workspace/locbench/IR-based/models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --repos_root /workspace/locbench/repos/locbench_repos \
  --gpu_id 0 \
  --batch_size 16

# SFR 评估 - 使用 run_with_index_sfr.py
python method/dense/run_with_index_sfr.py \
  --index_dir /workspace/locbench/IR-based/new_index_data/ir_function_SFR-Embedding-Code-2B_R/dense_index_ir_function \
  --dataset_path /workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --output_folder /workspace/locbench/IR-based/new_outputs/sfr_eval \
  --model_name /workspace/locbench/IR-based/models/hf_bundle/Salesforce__SFR-Embedding-Code-2B_R \
  --trust_remote_code \
  --mapper_type ast \
  --repos_root /workspace/locbench/repos/locbench_repos \
  --gpu_id 0 \
  --batch_size 16