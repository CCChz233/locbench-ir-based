import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def _best_text(instance: dict) -> str:
    """Pick a query string from an instance."""
    for key in ("problem_statement", "issue", "description", "prompt", "text"):
        val = instance.get(key)
        if val:
            return val
    return ""


def _tokenize(text: str) -> set[str]:
    """Simple whitespace/regex tokenizer for Jaccard."""
    tokens = re.split(r"\W+", text.lower())
    return set(t for t in tokens if t)


class CodeBlock:
    def __init__(self, file_path: str, description: str, code_content: str, start_line: int, end_line: int):
        self.file_path = file_path
        self.description = description
        self.code_content = code_content
        self.start_line = start_line
        self.end_line = end_line


class BlockDataset(Dataset):
    def __init__(self, blocks: List[CodeBlock], tokenizer: AutoTokenizer, max_length: int):
        self.blocks = blocks
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int):
        block = self.blocks[idx]
        text = f"{block.description}\n\n{block.code_content}"
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)


def split_file(path: Path, repo_root: Path, block_size: int, max_blocks: int | None) -> List[CodeBlock]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    blocks: List[CodeBlock] = []
    start = 0
    while start < len(lines):
        end = min(start + block_size, len(lines))
        chunk = lines[start:end]
        if any(line.strip() for line in chunk):
            rel_path = str(path.relative_to(repo_root))
            desc = f"file path: {rel_path}\nlines: {start}-{end - 1}"
            blocks.append(
                CodeBlock(
                    file_path=rel_path,
                    description=desc,
                    code_content="\n".join(chunk),
                    start_line=start,
                    end_line=end - 1,
                )
            )
        start = end
        if max_blocks and len(blocks) >= max_blocks:
            break
    return blocks


def collect_blocks(repo_path: str, block_size: int, max_blocks_per_file: int | None) -> List[CodeBlock]:
    repo_root = Path(repo_path)
    blocks: List[CodeBlock] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".py", ".java", ".js", ".ts", ".go", ".rs", ".cpp", ".c", ".hpp", ".h"}:
            continue
        blocks.extend(split_file(path, repo_root, block_size, max_blocks_per_file))
    if not blocks:
        raise RuntimeError(f"No source files found under {repo_path}")
    return blocks


def embed_texts(
    texts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    class TextDataset(Dataset):
        def __init__(self, items: List[str]):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx: int):
            encoded = tokenizer(
                self.items[idx],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    ds = TextDataset(texts)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for input_ids, attn_mask in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            token_embeddings = outputs[0]
            mask = attn_mask.unsqueeze(-1)
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            sent_emb = summed / counts
            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            outs.append(sent_emb.cpu())
    return torch.cat(outs, dim=0)


def embed_blocks(
    blocks: List[CodeBlock],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    ds = BlockDataset(blocks, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for input_ids, attn_mask in tqdm(loader, desc="Embedding blocks"):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            token_embeddings = outputs[0]
            mask = attn_mask.unsqueeze(-1)
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            sent_emb = summed / counts
            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            outs.append(sent_emb.cpu())
    return torch.cat(outs, dim=0)


def rank_files(
    block_scores: List[Tuple[int, float]],
    blocks: List[CodeBlock],
    top_k_files: int,
) -> List[str]:
    file_scores: Dict[str, float] = {}
    for idx, score in block_scores:
        f = blocks[idx].file_path
        file_scores[f] = file_scores.get(f, 0.0) + float(score)
    ranked = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:top_k_files]]


def _instance_id_to_repo_name(instance_id: str) -> str:
    """Convert instance_id (e.g., 'UXARRAY__uxarray-1117') to repo folder name (e.g., 'UXARRAY_uxarray')."""
    repo_part = re.sub(r"-\d+$", "", instance_id)
    return repo_part.replace("__", "_")


def run(args: argparse.Namespace) -> None:
    if not args.repo_path and not args.repos_root:
        raise ValueError("Must provide --repo_path (single repo) or --repos_root (multi-repo).")

    use_dense = args.mode == "dense"
    if use_dense:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name).to(device)
    else:
        device = torch.device("cpu")
        tokenizer = None
        model = None

    with open(args.dataset_path, "r") as f:
        instances = [json.loads(line) for line in f]
    if args.eval_n_limit:
        instances = instances[: args.eval_n_limit]

    queries = [_best_text(ins) for ins in instances]
    if use_dense:
        query_embeddings = embed_texts(queries, model, tokenizer, args.max_length, args.batch_size, device)
    else:
        query_tokens = [_tokenize(q) for q in queries]

    # cache per-repo data
    repo_cache_dense: Dict[str, Tuple[List[CodeBlock], torch.Tensor, torch.Tensor]] = {}
    repo_cache_sparse: Dict[str, Tuple[List[CodeBlock], List[set[str]]]] = {}

    def get_repo_embeddings(repo_dir: str) -> Tuple[List[CodeBlock], torch.Tensor, torch.Tensor]:
        if repo_dir in repo_cache_dense:
            return repo_cache_dense[repo_dir]
        blocks = collect_blocks(repo_dir, args.block_size, args.max_blocks_per_file)
        block_emb = embed_blocks(blocks, model, tokenizer, args.max_length, args.batch_size, device)
        block_emb_t = block_emb.t()
        repo_cache_dense[repo_dir] = (blocks, block_emb, block_emb_t)
        return repo_cache_dense[repo_dir]

    def get_repo_tokens(repo_dir: str) -> Tuple[List[CodeBlock], List[set[str]]]:
        if repo_dir in repo_cache_sparse:
            return repo_cache_sparse[repo_dir]
        blocks = collect_blocks(repo_dir, args.block_size, args.max_blocks_per_file)
        block_tokens = [_tokenize(f"{b.description}\n{b.code_content}") for b in blocks]
        repo_cache_sparse[repo_dir] = (blocks, block_tokens)
        return repo_cache_sparse[repo_dir]

    os.makedirs(args.output_folder, exist_ok=True)
    output_file = os.path.join(args.output_folder, "loc_outputs.jsonl")
    with open(output_file, "w") as fout, torch.no_grad():
        iterator = zip(instances, query_embeddings) if use_dense else zip(instances, query_tokens)
        for ins, q in tqdm(iterator, total=len(instances), desc="Ranking"):
            instance_id = ins.get("instance_id", "")
            if args.repos_root:
                repo_name = _instance_id_to_repo_name(instance_id)
                repo_dir = os.path.join(args.repos_root, repo_name)
            else:
                repo_dir = args.repo_path

            if not os.path.isdir(repo_dir):
                # Skip if repo not found
                record = {
                    "instance_id": instance_id,
                    "found_files": [],
                    "found_modules": [],
                    "found_entities": [],
                    "raw_output_loc": [],
                }
                fout.write(json.dumps(record) + "\n")
                continue

            if use_dense:
                blocks, _, block_emb_t = get_repo_embeddings(repo_dir)
                scores = torch.matmul(q, block_emb_t)  # (B,)
                topk = min(args.top_k_blocks, scores.numel())
                if topk == 0:
                    found_files = []
                else:
                    topk_scores, topk_idx = torch.topk(scores, k=topk)
                    block_scores = list(zip(topk_idx.tolist(), topk_scores.tolist()))
                    found_files = rank_files(block_scores, blocks, args.top_k_files)
            else:
                blocks, block_tokens = get_repo_tokens(repo_dir)
                scores_list: List[Tuple[int, float]] = []
                for idx, bt in enumerate(block_tokens):
                    if not bt:
                        continue
                    inter = len(q & bt)
                    union = len(q | bt)
                    score = inter / union if union else 0.0
                    scores_list.append((idx, score))
                scores_list.sort(key=lambda x: x[1], reverse=True)
                block_scores = scores_list[: args.top_k_blocks]
                found_files = rank_files(block_scores, blocks, args.top_k_files)
            record = {
                "instance_id": instance_id,
                "found_files": found_files,
                "found_modules": [],
                "found_entities": [],
                "raw_output_loc": [],
            }
            fout.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RepoCoder locator (dense retrieval via nov3630/RLRetriever).")
    parser.add_argument("--repo_path", help="Path to a single repository root (used if repos_root not provided).")
    parser.add_argument("--repos_root", help="Root directory containing multiple repos (derived from instance_id).")
    parser.add_argument("--dataset_path", required=True, help="Local JSONL dataset with instance_id and problem_statement.")
    parser.add_argument("--output_folder", required=True, help="Output directory for loc_outputs.jsonl.")
    parser.add_argument("--model_name", default="nov3630/RLRetriever", help="HF model id or local path.")
    parser.add_argument("--mode", choices=["dense", "jaccard"], default="dense", help="dense (UniXcoder) or jaccard (BoW).")
    parser.add_argument("--block_size", type=int, default=15, help="Lines per block.")
    parser.add_argument("--max_blocks_per_file", type=int, default=None, help="Optional cap per file to shrink index.")
    parser.add_argument("--max_length", type=int, default=512, help="Max tokens for encoding.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding.")
    parser.add_argument("--top_k_blocks", type=int, default=50, help="Top blocks per query before aggregating to files.")
    parser.add_argument("--top_k_files", type=int, default=10, help="Final file-level results.")
    parser.add_argument("--eval_n_limit", type=int, default=0, help="Optional cap on instances.")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
