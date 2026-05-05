"""CLI entry point for Tableau .hyper export and optional publish."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import GOLD_DIR, HYPER_DIR


def main():
    parser = argparse.ArgumentParser(description="Export Gold KPIs to Tableau .hyper")
    parser.add_argument("--publish", action="store_true", help="Also publish to Tableau Server")
    parser.add_argument("--refresh", action="store_true", help="Trigger extract refresh after publish")
    parser.add_argument("--sample", action="store_true", help="Also export Silver 10% sample")
    parser.add_argument("--workbook", action="store_true", help="Generate .twb workbook from template")
    parser.add_argument("--server-url", help="Tableau Server URL for workbook connections")
    parser.add_argument(
        "--export-view", metavar="VIEW_URL",
        help="Export a view as PDF (use --fmt to change format)",
    )
    parser.add_argument("--fmt", choices=["pdf", "png"], default="pdf")
    parser.add_argument("--output", type=Path, help="Output path for --export-view")
    args = parser.parse_args()

    from tableau.hyper_writer import gold_to_hyper, export_silver_sample

    hyper_paths = gold_to_hyper(GOLD_DIR, HYPER_DIR)
    print(f"\nExported {len(hyper_paths)} .hyper files to {HYPER_DIR}")

    if args.sample:
        sample_path = export_silver_sample()
        print(f"Silver sample → {sample_path}")

    if args.workbook:
        from tableau.workbook import generate_workbook
        twb_path = generate_workbook(
            hyper_dir=HYPER_DIR,
            server_url=args.server_url,
        )
        print(f"Workbook → {twb_path}")

    if args.publish:
        from tableau.publish import publish_all_hyper, refresh_datasource_tsc, _tsc_available

        results = publish_all_hyper(HYPER_DIR)
        print(f"Published {len(results)} datasources")

        if args.refresh and _tsc_available():
            for name, luid in results.items():
                if luid:
                    refresh_datasource_tsc(luid)

    if args.export_view:
        from tableau.publish import _tsc_available, _tabcmd_available

        output_path = args.output or Path(f"exports/{args.export_view.replace('/', '_')}.{args.fmt}")
        if _tsc_available():
            from tableau.publish import export_view_tsc
            export_view_tsc(args.export_view, output_path, fmt=args.fmt)
        elif _tabcmd_available():
            from tableau.publish import export_view_tabcmd
            export_view_tabcmd(args.export_view, output_path, fmt=args.fmt)
        else:
            print("ERROR: No Tableau client available for view export")
            sys.exit(1)


if __name__ == "__main__":
    main()
