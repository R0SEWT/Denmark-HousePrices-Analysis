"""CLI entry point for Gold layer KPI computation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medallion.gold import run_gold


def main():
    manifest = run_gold()
    print("\nGold KPI summary:")
    for kpi_name, info in manifest["kpis"].items():
        rows_key = "rows" if "rows" in info else "episodes"
        print(f"  {kpi_name}: {info[rows_key]} {rows_key} → {info['file']}")


if __name__ == "__main__":
    main()
