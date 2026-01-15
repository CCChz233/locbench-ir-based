import argparse
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json


class CodeWindow:
    def __init__(self, file_path: str, context: str, start_line: int, end_line: int):
        self.file_path = file_path
        self.context = context
        self.start_line = start_line
        self.end_line = end_line


def iterate_files(repo_root: Path) -> List[Path]:
    return [p for p in repo_root.rglob("*") if p.is_file() and p.suffix.lower() in {
        ".py", ".java", ".js", ".ts", ".go", ".rs", ".cpp", ".c", ".hpp", ".h"
    }]


def sliding_windows(code: str, window_size: int, slice_size: int) -> List[Dict[str, Any]]:
    lines = code.splitlines()
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    step = max(1, window_size // slice_size) if slice_size > 0 else window_size
    delta = window_size // 2
    windows = []
    for line_no in range(0, len(lines), step):
        start_line = max(0, line_no - delta)
        end_line = min(len(lines), line_no + window_size - delta)
        window_lines = lines[start_line:end_line]
        if not any(l.strip() for l in window_lines):
            continue
        windows.append((start_line, end_line, "\n".join(window_lines)))
    return windows


def collect_windows(repo_path: str, window_size: int, slice_size: int) -> List[CodeWindow]:
    repo_root = Path(repo_path)
    windows: List[CodeWindow] = []
    for f in iterate_files(repo_root):
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="utf-8", errors="ignore")
        rel = str(f.relative_to(repo_root))
        for start, end, ctx in sliding_windows(text, window_size, slice_size):
            windows.append(CodeWindow(rel, ctx, start, end - 1))
    if not windows:
        raise RuntimeError(f"No windows found under {repo_path}")
    return windows


class WindowDataset(Dataset):
    def __init__(self, windows: List[CodeWindow], tokenizer: AutoTokenizer, max_length: int):
        self.windows = windows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        text = f"file path: {w.file_path}\nlines: {w.start_line}-{w.end_line}\n\n{w.context}"
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)


def embed_windows(windows: List[CodeWindow], model: AutoModel, tokenizer: AutoTokenizer,
                  max_length: int, batch_size: int, device: torch.device) -> torch.Tensor:
    ds = WindowDataset(windows, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    outs = []
    with torch.no_grad():
        for input_ids, attn_mask in tqdm(loader, desc="Embedding windows"):
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


def save_index(output_dir: Path, embeddings: torch.Tensor, windows: List[CodeWindow]):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_dir / "embeddings.pt")
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for idx, w in enumerate(windows):
            rec = {
                "block_id": idx,
                "file_path": w.file_path,
                "start_line": w.start_line,
                "end_line": w.end_line,
            }
            f.write(json.dumps(rec) + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Build sliding-window dense index (RepoCoder-style).")
    ap.add_argument("--repo_path", required=True, help="Repository root to index.")
    ap.add_argument("--output_dir", required=True, help="Where to save embeddings + metadata.")
    ap.add_argument("--model_name", default="nov3630/RLRetriever", help="HF model id or local path.")
    ap.add_argument("--window_size", type=int, default=20, help="Sliding window size (lines).")
    ap.add_argument("--slice_size", type=int, default=2, help="Slice size divisor; step = window_size/slice_size.")
    ap.add_argument("--max_length", type=int, default=512, help="Max tokens per window.")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding.")
    ap.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    windows = collect_windows(args.repo_path, args.window_size, args.slice_size)
    embeddings = embed_windows(windows, model, tokenizer, args.max_length, args.batch_size, device)
    save_index(Path(args.output_dir), embeddings, windows)


if __name__ == "__main__":
    main()
