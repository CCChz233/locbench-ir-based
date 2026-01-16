python method/index/batch_build_index.py \
    --repo_path /Users/chz/code/locbench/locbench_repos \
    --index_dir /tmp/ir_index_test \
    --model_name /Users/chz/code/locbench/locbench-ir-based/models/CodeRankEmbed\
    --strategy fixed \
    --block_size 15 \
    --num_processes 1 \
    --force_cpu \
    --trust_remote_code

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "LLukas22/CodeRankEmbed",
    local_dir="/Users/chz/code/locbench/locbench-ir-based/models/CodeRankEmbed",
    local_dir_use_symlinks=False,
    resume_download=True,
  )
PY
