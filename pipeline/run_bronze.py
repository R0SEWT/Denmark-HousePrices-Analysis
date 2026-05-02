"""CLI entry point for Bronze layer ingestion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medallion.bronze import run_bronze


def main():
    manifest = run_bronze()
    print(f"Bronze manifest: {manifest['output_row_count']} rows, "
          f"{len(manifest['columns'])} columns")


if __name__ == "__main__":
    main()
