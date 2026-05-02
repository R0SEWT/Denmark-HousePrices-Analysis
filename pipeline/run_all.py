"""Orchestrator: Bronze -> Silver -> Gold -> Tableau in sequence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medallion.bronze import run_bronze
from medallion.silver import run_silver
from medallion.gold import run_gold


def main():
    print("=" * 60)
    print("MEDALLION PIPELINE: Bronze → Silver → Gold")
    print("=" * 60)

    print("\n── BRONZE ──")
    bronze_manifest = run_bronze()

    print("\n── SILVER ──")
    silver_manifest = run_silver(bronze_manifest)

    print("\n── GOLD ──")
    gold_manifest = run_gold(silver_manifest)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Bronze: {bronze_manifest['output_row_count']:,} rows")
    print(f"  Silver: {silver_manifest['output_row_count']:,} rows")
    print(f"  Gold: {len(gold_manifest['kpis'])} KPIs")
    print("=" * 60)

    try:
        from tableau.hyper_writer import gold_to_hyper
        from config import GOLD_DIR, HYPER_DIR
        print("\n── TABLEAU ──")
        hyper_paths = gold_to_hyper(GOLD_DIR, HYPER_DIR)
        print(f"  {len(hyper_paths)} .hyper files exported")
    except ImportError:
        print("\n[SKIP] Tableau export: tableauhyperapi not installed")


if __name__ == "__main__":
    main()
