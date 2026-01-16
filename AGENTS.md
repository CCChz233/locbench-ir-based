# Repository Guidelines

## Project Structure & Module Organization
- `method/`: retrieval methods (bm25, dense, jaccard), mappers, and index builders; add new approaches under `method/<name>/`.
- `repo_index/`: code parsing and index abstractions used by build/run scripts.
- `util/` and `plugins/`: shared helpers, prompts, and plugin utilities.
- `scripts/`: runnable baselines and benchmarks.
- `models/`: model configs/tokenizers; large weights are ignored by `.gitignore`.
- `evaluation/`: metrics and notebook; `index_data/` and `outputs/` are generated artifacts.

## Build, Test, and Development Commands
- `conda activate locagent` then `export PYTHONPATH="$(pwd):../LocAgent:$PYTHONPATH"` and `export TOKENIZERS_PARALLELISM=false`.
- `python build_bm25_index.py --dataset czlll/Loc-Bench_V1 --split test --repo_path ../repos/locbench_repos --index_dir index_data` builds the BM25 index.
- `python method/index/batch_build_index.py --repo_path ../repos/locbench_repos --index_dir index_data --model_name models/rlretriever --strategy fixed --block_size 15` builds a dense index.
- `python scripts/run_bm25_baseline.py ...`, `python method/dense/run_with_index.py ...`, or `python method/RepoCoder/run_locator.py --mode jaccard ...` run retrieval.
- `python -c "from evaluation.eval_metric import evaluate_results; ..."` scores a `loc_outputs.jsonl`.

## Coding Style & Naming Conventions
- Python, 4-space indentation; follow the existing PEP8-like style.
- Use `snake_case` for functions/modules, `PascalCase` for classes, and `CONSTANT_CASE` for module constants.
- Keep the output schema stable: `loc_outputs.jsonl` with `instance_id`, `found_files`, `found_modules`, `found_entities`, `raw_output_loc`.

## Testing Guidelines
- Tests are script-style and run directly with Python.
- `python method/test_refactoring.py` exercises mappers; set `GRAPH_INDEX_DIR` and `BM25_INDEX_DIR` if auto-detection fails.
- `python method/index/test_span_ids.py`, `python method/test_mapper_switching.py`, and `python test_missing_repos.py` cover focused checks.
- No formal coverage target noted; keep tests small and dataset-aware.

## Commit & Pull Request Guidelines
- History uses short, sentence-case messages (e.g., "Exclude large data files..."); keep commits concise and descriptive.
- PRs should describe algorithm changes, expected data/index paths, and provide a runnable command or sample output.
- Do not commit generated artifacts (`outputs/`, `index_data/`, model weights, datasets, images); rely on `.gitignore`.

## Configuration & Data Notes
- This repo expects a sibling `../LocAgent` for shared modules and index data.
- Common data roots: `../data/Loc-Bench_V1_dataset.jsonl`, `../repos/locbench_repos`, and `playground/locbench_repos`.
