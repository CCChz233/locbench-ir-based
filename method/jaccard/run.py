"""
Run Jaccard/BoW locator (non-overlapping blocks, on-the-fly tokenization).
"""
import sys
from method.RepoCoder import run_locator


if __name__ == "__main__":
    sys.argv.extend(["--mode", "jaccard"])
    sys.exit(run_locator.run(run_locator.parse_args()))
