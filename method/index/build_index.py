"""
Unified dense index builder with multiple block strategies.
- fixed: non-overlapping blocks of N lines (default 15)
- sliding: overlapping window (window_size, slice_size controls step)
- rl_fixed: RLCoder fixed_block (12 non-empty lines, max 5000 lines)
- rl_mini: RLCoder mini_block (empty-line segments, stitched to <=15 lines, max 5000 lines)

Outputs: embeddings.pt + metadata.jsonl
"""

import argparse
import json
import pickle
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from dependency_graph import RepoEntitySearcher
from dependency_graph.build_graph import NODE_TYPE_FUNCTION, NODE_TYPE_CLASS


class Block:
    def __init__(self, file_path: str, start: int, end: int, content: str, block_type: str):
        self.file_path = file_path
        self.start = start
        self.end = end
        self.content = content
        self.block_type = block_type


def iter_files(repo_root: Path) -> List[Path]:
    return [
        p for p in repo_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".py", ".java", ".js", ".ts", ".go", ".rs", ".cpp", ".c", ".hpp", ".h"}
    ]


def blocks_fixed_lines(text: str, rel: str, block_size: int) -> List[Block]:
    lines = text.splitlines()
    blocks: List[Block] = []
    start = 0
    while start < len(lines):
        end = min(start + block_size, len(lines))
        chunk = lines[start:end]
        if any(l.strip() for l in chunk):
            blocks.append(Block(rel, start, end - 1, "\n".join(chunk), "fixed"))
        start = end
    return blocks


def blocks_sliding(text: str, rel: str, window_size: int, slice_size: int) -> List[Block]:
    lines = text.splitlines()
    delta = window_size // 2
    step = max(1, window_size // slice_size) if slice_size > 0 else window_size
    blocks: List[Block] = []
    for line_no in range(0, len(lines), step):
        start = max(0, line_no - delta)
        end = min(len(lines), line_no + window_size - delta)
        chunk = lines[start:end]
        if any(l.strip() for l in chunk):
            blocks.append(Block(rel, start, end - 1, "\n".join(chunk), "sliding"))
    return blocks


def blocks_rl_fixed(text: str, rel: str, max_lines: int = 5000) -> List[Block]:
    lines = [line for line in text.split("\n") if line.strip()]
    blocks: List[Block] = []
    for i in range(0, min(len(lines), max_lines), 12):
        start = i
        end = min(i + 12, len(lines))
        chunk = lines[start:end]
        blocks.append(Block(rel, start, end - 1, "\n".join(chunk), "rl_fixed"))
    return blocks


def blocks_rl_mini(text: str, rel: str, max_lines: int = 5000) -> List[Block]:
    mini_blocks = []
    cur = []
    for line in text.splitlines():
        if line.strip():
            cur.append(line)
        else:
            if cur:
                mini_blocks.append(cur)
                cur = []
    if cur:
        mini_blocks.append(cur)

    temp = []
    for mb in mini_blocks:
        if len(mb) > 15:
            for idx in range(0, len(mb), 15):
                temp.append(mb[idx: idx + 15])
        else:
            temp.append(mb)
    mini_blocks = temp

    blocks: List[Block] = []
    current = []
    total = 0
    for block in mini_blocks:
        if total >= max_lines:
            break
        if len(current) + len(block) <= 15:
            current.extend(block)
            total += len(block)
        else:
            if current:
                blocks.append(Block(rel, total - len(current) + 1, total, "\n".join(current), "rl_mini"))
            current = block
            total += len(block)
    if current:
        blocks.append(Block(rel, total - len(current) + 1, total, "\n".join(current), "rl_mini"))
    return blocks


def collect_blocks(repo_path: str, strategy: str, block_size: int, window_size: int, slice_size: int) -> List[Block]:
    repo_root = Path(repo_path)
    blocks: List[Block] = []
    for p in iter_files(repo_root):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="utf-8", errors="ignore")
        rel = str(p.relative_to(repo_root))
        if strategy == "fixed":
            blocks.extend(blocks_fixed_lines(text, rel, block_size))
        elif strategy == "sliding":
            blocks.extend(blocks_sliding(text, rel, window_size, slice_size))
        elif strategy == "rl_fixed":
            blocks.extend(blocks_rl_fixed(text, rel))
        elif strategy == "rl_mini":
            blocks.extend(blocks_rl_mini(text, rel))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    if not blocks:
        raise RuntimeError(f"No blocks produced under {repo_path}")
    return blocks


class BlockDataset(Dataset):
    def __init__(self, blocks: List[Block], tokenizer: AutoTokenizer, max_length: int):
        self.blocks = blocks
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx: int):
        b = self.blocks[idx]
        text = f"file path: {b.file_path}\nlines: {b.start}-{b.end}\n\n{b.content}"
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


def embed_blocks(blocks: List[Block], model: AutoModel, tokenizer: AutoTokenizer, max_length: int, batch_size: int, device: torch.device) -> torch.Tensor:
    ds = BlockDataset(blocks, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    outs = []
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


def extract_span_ids_from_graph(
    blocks: List[Block],
    graph_index_file: str,
    repo_path: str
) -> Dict[int, List[str]]:
    """
    从Graph索引中为代码块提取span_ids
    
    Args:
        blocks: 代码块列表
        graph_index_file: Graph索引文件路径（.pkl）
        repo_path: 仓库根目录（用于路径匹配）
    
    Returns:
        Dict[int, List[str]]: 代码块索引 -> span_ids列表的映射
    """
    if not os.path.exists(graph_index_file):
        print(f"Warning: Graph index file not found: {graph_index_file}")
        return {}
    
    # 加载Graph索引
    try:
        with open(graph_index_file, 'rb') as f:
            graph = pickle.load(f)
        searcher = RepoEntitySearcher(graph)
    except Exception as e:
        print(f"Warning: Failed to load graph index {graph_index_file}: {e}")
        return {}
    
    # 按文件分组代码块
    blocks_by_file = {}
    for idx, block in enumerate(blocks):
        file_path = block.file_path
        if file_path not in blocks_by_file:
            blocks_by_file[file_path] = []
        blocks_by_file[file_path].append((idx, block))
    
    # 为每个代码块查找span_ids
    span_ids_map = {}
    
    for file_path, file_blocks in blocks_by_file.items():
        # 查找该文件的所有节点（函数/类）
        file_nodes = []
        for node_id in searcher.G:
            if node_id.startswith(f"{file_path}:"):
                file_nodes.append(node_id)
        
        # 为每个代码块查找重叠的节点
        for block_idx, block in file_blocks:
            span_ids = set()
            block_start_1based = block.start + 1  # 转换为1-based
            block_end_1based = block.end + 1
            
            for node_id in file_nodes:
                if not searcher.has_node(node_id):
                    continue
                
                try:
                    node_data = searcher.get_node_data([node_id], return_code_content=False)[0]
                    node_start = node_data.get("start_line")
                    node_end = node_data.get("end_line")
                    
                    if node_start is None or node_end is None:
                        continue
                    
                    # 检查行号范围是否重叠
                    if not (block_end_1based < node_start or block_start_1based > node_end):
                        node_type = node_data.get("type")
                        if node_type in [NODE_TYPE_FUNCTION, NODE_TYPE_CLASS]:
                            # 提取实体名称（node_id格式: file_path:entity_name）
                            if ":" in node_id:
                                entity_name = node_id.split(":", 1)[1]
                                span_ids.add(entity_name)
                except Exception as e:
                    # 跳过有问题的节点
                    continue
            
            if span_ids:
                span_ids_map[block_idx] = list(span_ids)  # 去重并转为列表
    
    return span_ids_map


def save_index(
    output_dir: Path,
    embeddings: torch.Tensor,
    blocks: List[Block],
    strategy: str,
    span_ids_map: Optional[Dict[int, List[str]]] = None
):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_dir / "embeddings.pt")
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for idx, b in enumerate(blocks):
            metadata = {
                "block_id": idx,
                "file_path": b.file_path,
                "start_line": b.start,
                "end_line": b.end,
                "block_type": b.block_type,
                "strategy": strategy,
            }
            # 如果有span_ids，添加到metadata
            if span_ids_map and idx in span_ids_map:
                metadata["span_ids"] = span_ids_map[idx]
            
            f.write(json.dumps(metadata) + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Build dense index with multiple block strategies.")
    ap.add_argument("--repo_path", required=True, help="Repository root to index.")
    ap.add_argument("--output_dir", required=True, help="Where to save embeddings + metadata.")
    ap.add_argument("--model_name", default="nov3630/RLRetriever", help="HF model id or local path.")
    ap.add_argument("--strategy", choices=["fixed", "sliding", "rl_fixed", "rl_mini"], default="fixed")
    ap.add_argument("--block_size", type=int, default=15, help="Used for strategy=fixed.")
    ap.add_argument("--window_size", type=int, default=20, help="Used for strategy=sliding.")
    ap.add_argument("--slice_size", type=int, default=2, help="Used for strategy=sliding.")
    ap.add_argument("--max_length", type=int, default=512, help="Max tokens for encoding.")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding.")
    ap.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA is available.")
    ap.add_argument(
        "--graph_index_file",
        type=str,
        default=None,
        help="Graph索引文件路径（.pkl），如果提供将在metadata中添加span_ids"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    blocks = collect_blocks(args.repo_path, args.strategy, args.block_size, args.window_size, args.slice_size)
    embeddings = embed_blocks(blocks, model, tokenizer, args.max_length, args.batch_size, device)
    
    # 如果提供了Graph索引，提取span_ids
    span_ids_map = {}
    if args.graph_index_file:
        print(f"Extracting span_ids from graph index: {args.graph_index_file}")
        span_ids_map = extract_span_ids_from_graph(blocks, args.graph_index_file, args.repo_path)
        print(f"Extracted span_ids for {len(span_ids_map)}/{len(blocks)} blocks")
    
    save_index(Path(args.output_dir), embeddings, blocks, args.strategy, span_ids_map)


if __name__ == "__main__":
    main()
