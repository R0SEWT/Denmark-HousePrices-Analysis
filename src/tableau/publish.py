"""Publish .hyper extracts to Tableau Server via REST API or tabcmd."""

import os
from pathlib import Path


def publish_hyper_rest(
    hyper_path: Path,
    server_url: str | None = None,
    token_name: str | None = None,
    token_value: str | None = None,
    site_id: str = "",
    project_name: str = "Default",
    overwrite: bool = True,
) -> str:
    """Publish a .hyper file to Tableau Server using tableauserverclient.

    Credentials from env vars: TABLEAU_SERVER_URL, TABLEAU_TOKEN_NAME,
    TABLEAU_TOKEN_VALUE, TABLEAU_SITE_ID, TABLEAU_PROJECT.

    Returns the datasource LUID on success.
    """
    import tableauserverclient as TSC

    server_url = server_url or os.environ["TABLEAU_SERVER_URL"]
    token_name = token_name or os.environ["TABLEAU_TOKEN_NAME"]
    token_value = token_value or os.environ["TABLEAU_TOKEN_VALUE"]
    site_id = site_id or os.environ.get("TABLEAU_SITE_ID", "")
    project_name = project_name or os.environ.get("TABLEAU_PROJECT", "Default")

    auth = TSC.PersonalAccessTokenAuth(token_name, token_value, site_id)
    server = TSC.Server(server_url, use_server_version=True)

    with server.auth.sign_in(auth):
        projects, _ = server.projects.get()
        target_project = next(
            (p for p in projects if p.name == project_name), None
        )
        if target_project is None:
            raise ValueError(f"Project '{project_name}' not found on server")

        publish_mode = (
            TSC.Server.PublishMode.Overwrite
            if overwrite
            else TSC.Server.PublishMode.CreateNew
        )

        ds = TSC.DatasourceItem(target_project.id)
        ds = server.datasources.publish(ds, str(hyper_path), publish_mode)

        print(f"Published {hyper_path.name} → {server_url} (LUID: {ds.id})")
        return ds.id


def publish_all_hyper(
    hyper_dir: Path,
    **kwargs,
) -> dict[str, str]:
    """Publish all .hyper files in a directory. Returns {filename: luid}."""
    results = {}
    for hyper_path in sorted(hyper_dir.glob("*.hyper")):
        luid = publish_hyper_rest(hyper_path, **kwargs)
        results[hyper_path.name] = luid
    return results
