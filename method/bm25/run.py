"""
Thin wrapper to run the BM25 baseline with standardized arguments.
Relies on pre-built graph/BM25 indexes.
"""

import sys
import scripts.run_bm25_baseline as bm25_main


if __name__ == "__main__":
    # Delegate to existing entrypoint
    sys.exit(bm25_main.main())
