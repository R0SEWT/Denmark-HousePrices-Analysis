"""Publish .hyper extracts to Tableau Server via TSC REST API or tabcmd 2.0."""

import os
import shutil
import subprocess
from pathlib import Path


def _get_credentials() -> dict[str, str]:
    return {
        "server_url": os.environ["TABLEAU_SERVER_URL"],
        "token_name": os.environ["TABLEAU_PAT_NAME"],
        "token_value": os.environ["TABLEAU_PAT_SECRET"],
        "site_id": os.environ.get("TABLEAU_SITE_ID", ""),
        "project_name": os.environ.get("TABLEAU_PROJECT", "Default"),
    }


def _tsc_available() -> bool:
    try:
        import tableauserverclient  # noqa: F401
        return True
    except ImportError:
        return False


def _tabcmd_available() -> bool:
    return shutil.which("tabcmd") is not None


# ── TSC (REST API) ────────────────────────────────────────────────────


def publish_hyper_tsc(
    hyper_path: Path,
    overwrite: bool = True,
    **cred_overrides: str,
) -> str:
    """Publish a .hyper file via tableauserverclient. Returns datasource LUID."""
    import tableauserverclient as TSC

    creds = {**_get_credentials(), **cred_overrides}

    auth = TSC.PersonalAccessTokenAuth(
        creds["token_name"], creds["token_value"], creds["site_id"],
    )
    server = TSC.Server(creds["server_url"], use_server_version=True)

    with server.auth.sign_in(auth):
        projects, _ = server.projects.get()
        target_project = next(
            (p for p in projects if p.name == creds["project_name"]), None,
        )
        if target_project is None:
            raise ValueError(f"Project '{creds['project_name']}' not found on server")

        publish_mode = (
            TSC.Server.PublishMode.Overwrite
            if overwrite
            else TSC.Server.PublishMode.CreateNew
        )

        ds = TSC.DatasourceItem(target_project.id)
        ds = server.datasources.publish(ds, str(hyper_path), publish_mode)

        print(f"[TSC] Published {hyper_path.name} → {creds['server_url']} (LUID: {ds.id})")
        return ds.id


def refresh_datasource_tsc(datasource_luid: str, **cred_overrides: str) -> None:
    """Trigger an extract refresh for a published datasource."""
    import tableauserverclient as TSC

    creds = {**_get_credentials(), **cred_overrides}
    auth = TSC.PersonalAccessTokenAuth(
        creds["token_name"], creds["token_value"], creds["site_id"],
    )
    server = TSC.Server(creds["server_url"], use_server_version=True)

    with server.auth.sign_in(auth):
        ds = server.datasources.get_by_id(datasource_luid)
        server.datasources.refresh(ds)
        print(f"[TSC] Refresh triggered for datasource {datasource_luid}")


def export_view_tsc(
    view_name: str,
    output_path: Path,
    fmt: str = "pdf",
    **cred_overrides: str,
) -> Path:
    """Export a Tableau view as PDF or PNG via TSC.

    `view_name` should match the view's content_url on the server.
    `fmt` must be 'pdf' or 'png'.
    """
    import tableauserverclient as TSC

    creds = {**_get_credentials(), **cred_overrides}
    auth = TSC.PersonalAccessTokenAuth(
        creds["token_name"], creds["token_value"], creds["site_id"],
    )
    server = TSC.Server(creds["server_url"], use_server_version=True)

    with server.auth.sign_in(auth):
        all_views, _ = server.views.get()
        target = next((v for v in all_views if v.content_url == view_name), None)
        if target is None:
            raise ValueError(f"View '{view_name}' not found on server")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "pdf":
            server.views.populate_pdf(target)
            with open(output_path, "wb") as f:
                f.write(target.pdf)
        elif fmt == "png":
            server.views.populate_image(target)
            with open(output_path, "wb") as f:
                f.write(target.image)
        else:
            raise ValueError(f"Unsupported format '{fmt}', use 'pdf' or 'png'")

        print(f"[TSC] Exported {view_name} → {output_path}")
        return output_path


# ── tabcmd 2.0 fallback ──────────────────────────────────────────────


def _run_tabcmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a tabcmd command with standard auth flags from env vars."""
    creds = _get_credentials()
    auth_args = [
        "--server", creds["server_url"],
        "--token-name", creds["token_name"],
        "--token-value", creds["token_value"],
    ]
    if creds["site_id"]:
        auth_args += ["--site", creds["site_id"]]

    cmd = ["tabcmd"] + args + auth_args
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result


def publish_hyper_tabcmd(
    hyper_path: Path,
    project_name: str | None = None,
    overwrite: bool = True,
) -> None:
    """Publish a .hyper file via tabcmd 2.0."""
    project_name = project_name or os.environ.get("TABLEAU_PROJECT", "Default")
    args = ["publish", str(hyper_path), "--project", project_name]
    if overwrite:
        args.append("--overwrite")
    _run_tabcmd(args)
    print(f"[tabcmd] Published {hyper_path.name} → project '{project_name}'")


def export_view_tabcmd(
    view_url: str,
    output_path: Path,
    fmt: str = "pdf",
) -> Path:
    """Export a view via tabcmd. `view_url` is the server-relative path."""
    if fmt == "pdf":
        args = ["export", view_url, "--pdf", "-f", str(output_path)]
    elif fmt == "png":
        args = ["export", view_url, "--png", "-f", str(output_path)]
    else:
        raise ValueError(f"Unsupported format '{fmt}', use 'pdf' or 'png'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_tabcmd(args)
    print(f"[tabcmd] Exported {view_url} → {output_path}")
    return output_path


# ── Unified entry points ─────────────────────────────────────────────


def publish_hyper(hyper_path: Path, overwrite: bool = True, **kwargs: str) -> str | None:
    """Publish a .hyper file. Tries TSC first, falls back to tabcmd."""
    if _tsc_available():
        return publish_hyper_tsc(hyper_path, overwrite=overwrite, **kwargs)
    if _tabcmd_available():
        publish_hyper_tabcmd(hyper_path, overwrite=overwrite)
        return None
    raise RuntimeError(
        "Neither tableauserverclient nor tabcmd found. "
        "Install with: pip install tableauserverclient  or  pip install tabcmd"
    )


def publish_all_hyper(hyper_dir: Path, **kwargs: str) -> dict[str, str | None]:
    """Publish all .hyper files in a directory. Returns {filename: luid_or_none}."""
    results = {}
    for hyper_path in sorted(hyper_dir.glob("*.hyper")):
        luid = publish_hyper(hyper_path, **kwargs)
        results[hyper_path.name] = luid
    return results
