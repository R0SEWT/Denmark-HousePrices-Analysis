"""CLI entry point for Silver layer enrichment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medallion.silver import run_silver


def main():
    manifest = run_silver()
    print(f"\nSilver manifest: {manifest['output_row_count']} rows, "
          f"geo coverage {manifest['geo_coverage_pct']}%, "
          f"STL regions {manifest['stl_regions']}")
    print("Flag counts:")
    for flag, count in manifest["flag_counts"].items():
        pct = count / manifest["output_row_count"] * 100
        print(f"  {flag}: {count:,} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
