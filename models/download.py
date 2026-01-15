#!/usr/bin/env python3
"""
使用 ModelScope 下载大模型到当前目录
模型列表：
  - Qwen/Qwen3-8B
  - Qwen/Qwen3-32B-Instruct
  - Qwen/Qwen2.5-7B-Instruct
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

用法：
  pip install modelscope
  python download_models.py
"""
import os
from pathlib import Path

try:
    from modelscope import snapshot_download
except ImportError:
    print("请先安装 modelscope: pip install modelscope")
    exit(1)

# 下载目标目录（当前文件夹）
DOWNLOAD_DIR = Path(__file__).resolve().parent

# 要下载的模型列表（ModelScope 仓库ID）
MODELS = [
    # "Qwen/Qwen3-8B",
    # "Qwen/Qwen3-32B",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    # "Qwen/Qwen2.5-32B-Instruct",
   "Salesforce/SFR-Embedding-Code-2B_R",
    # "Salesforce/SFR-Embedding-Code-400M_R",
    
    
]

def main():
    print(f"下载目录: {DOWNLOAD_DIR}")
    print(f"模型数量: {len(MODELS)}")
    print("-" * 50)

    for model_id in MODELS:
        # 子目录名：把 / 替换为 __
        local_name = model_id.replace("/", "__")
        local_dir = DOWNLOAD_DIR / local_name

        print(f"\n[下载] {model_id}")
        print(f"  目标: {local_dir}")

        try:
            path = snapshot_download(
                model_id=model_id,
                cache_dir=str(local_dir),
                revision="master",  # ModelScope 默认分支
            )
            print(f"  ✓ 完成: {path}")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            continue

    print("\n" + "=" * 50)
    print("全部下载任务完成！")
    print(f"模型保存在: {DOWNLOAD_DIR}")

if __name__ == "__main__":
    main()
