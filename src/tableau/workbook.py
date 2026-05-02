"""Tableau workbook (.twb) template management via XML patching."""

import xml.etree.ElementTree as ET
from pathlib import Path

from config import HYPER_DIR


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def patch_datasource_connections(
    twb_path: Path,
    output_path: Path | None = None,
    hyper_dir: Path = HYPER_DIR,
    server_url: str | None = None,
    site_id: str | None = None,
    project_name: str = "Default",
) -> Path:
    """Patch a .twb template to point at local .hyper files or published extracts.

    When `server_url` is provided, connections are rewritten to reference
    published datasources on the server. Otherwise, connections point to
    local .hyper file paths for offline development.

    Returns the path of the patched workbook.
    """
    output_path = output_path or twb_path.with_suffix(".patched.twb")

    tree = ET.parse(twb_path)
    root = tree.getroot()

    for ds in root.iter("datasource"):
        ds_name = ds.get("name", "")
        caption = ds.get("caption", ds_name)

        hyper_name = _resolve_hyper_name(caption, hyper_dir)
        if hyper_name is None:
            continue

        for conn in ds.iter("connection"):
            if server_url:
                conn.set("class", "tableauserverclient")
                conn.set("server", server_url)
                conn.set("dbname", hyper_name.replace(".hyper", ""))
                if site_id:
                    conn.set("site", site_id)
                conn.set("project", project_name)
            else:
                hyper_path = hyper_dir / hyper_name
                conn.set("class", "hyper")
                conn.set("dbname", str(hyper_path.resolve()))

    tree.write(output_path, encoding="unicode", xml_declaration=True)
    print(f"Patched workbook → {output_path}")
    return output_path


def _resolve_hyper_name(caption: str, hyper_dir: Path) -> str | None:
    """Match a datasource caption to a .hyper file in the directory."""
    normalized = caption.lower().replace(" ", "_").replace("-", "_")
    for hyper_path in hyper_dir.glob("*.hyper"):
        stem = hyper_path.stem.lower()
        if stem in normalized or normalized in stem:
            return hyper_path.name
    return None


def list_datasources(twb_path: Path) -> list[dict[str, str]]:
    """List datasource names and connection details in a .twb file."""
    tree = ET.parse(twb_path)
    root = tree.getroot()

    datasources = []
    for ds in root.iter("datasource"):
        name = ds.get("name", "")
        caption = ds.get("caption", name)
        connections = []
        for conn in ds.iter("connection"):
            connections.append({
                "class": conn.get("class", ""),
                "dbname": conn.get("dbname", ""),
                "server": conn.get("server", ""),
            })
        datasources.append({
            "name": name,
            "caption": caption,
            "connections": connections,
        })
    return datasources


def validate_template(twb_path: Path, hyper_dir: Path = HYPER_DIR) -> dict[str, bool]:
    """Check which datasources in a template have matching .hyper files."""
    datasources = list_datasources(twb_path)
    results = {}
    for ds in datasources:
        caption = ds["caption"]
        matched = _resolve_hyper_name(caption, hyper_dir) is not None
        results[caption] = matched
    return results
