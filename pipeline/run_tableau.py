"""CLI entry point for Tableau .hyper export and optional publish."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import GOLD_DIR, HYPER_DIR


def main():
    parser = argparse.ArgumentParser(description="Export Gold KPIs to Tableau .hyper")
    parser.add_argument("--publish", action="store_true", help="Also publish to Tableau Server")
    parser.add_argument("--sample", action="store_true", help="Also export Silver 10% sample")
    args = parser.parse_args()

    from tableau.hyper_writer import gold_to_hyper, export_silver_sample

    hyper_paths = gold_to_hyper(GOLD_DIR, HYPER_DIR)
    print(f"\nExported {len(hyper_paths)} .hyper files to {HYPER_DIR}")

    if args.sample:
        sample_path = export_silver_sample()
        print(f"Silver sample → {sample_path}")

    if args.publish:
        from tableau.publish import publish_all_hyper
        results = publish_all_hyper(HYPER_DIR)
        print(f"Published {len(results)} datasources")


if __name__ == "__main__":
    main()
