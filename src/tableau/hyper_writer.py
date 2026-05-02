"""Convert Gold parquet tables to Tableau .hyper extracts."""

from pathlib import Path

import pandas as pd

from config import GOLD_DIR, HYPER_DIR, SILVER_DIR


DTYPE_MAP = {
    "float64": "DOUBLE",
    "float32": "DOUBLE",
    "int64": "BIG_INT",
    "int32": "INTEGER",
    "Int16": "INTEGER",
    "Int64": "BIG_INT",
    "object": "TEXT",
    "str": "TEXT",
    "bool": "BOOL",
    "datetime64[ns]": "TIMESTAMP",
    "datetime64[ns, UTC]": "TIMESTAMP",
}


def write_dataframe_to_hyper(
    df: pd.DataFrame,
    hyper_path: Path,
    table_name: str,
    schema_name: str = "Extract",
) -> None:
    """Write a DataFrame to a .hyper file using tableauhyperapi."""
    from tableauhyperapi import (
        Connection,
        CreateMode,
        HyperProcess,
        Inserter,
        SqlType,
        TableDefinition,
        TableName,
        Telemetry,
    )

    type_map = {
        "DOUBLE": SqlType.double(),
        "BIG_INT": SqlType.big_int(),
        "INTEGER": SqlType.int_(),
        "TEXT": SqlType.text(),
        "BOOL": SqlType.bool_(),
        "TIMESTAMP": SqlType.timestamp(),
    }

    columns = []
    for col_name, dtype in df.dtypes.items():
        hyper_type_name = DTYPE_MAP.get(str(dtype), "TEXT")
        columns.append(
            TableDefinition.Column(col_name, type_map[hyper_type_name])
        )

    table_def = TableDefinition(
        table_name=TableName(schema_name, table_name),
        columns=columns,
    )

    hyper_path.parent.mkdir(parents=True, exist_ok=True)

    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(
            hyper.endpoint, str(hyper_path), CreateMode.CREATE_AND_REPLACE
        ) as conn:
            conn.catalog.create_schema_if_not_exists(schema_name)
            conn.catalog.create_table(table_def)

            with Inserter(conn, table_def) as inserter:
                for row in df.itertuples(index=False, name=None):
                    cleaned = [
                        None if pd.isna(v) else v
                        for v in row
                    ]
                    inserter.add_row(cleaned)
                inserter.execute()

    print(f"Hyper: {len(df)} rows → {hyper_path}")


def gold_to_hyper(
    gold_dir: Path = GOLD_DIR,
    hyper_dir: Path = HYPER_DIR,
    tables: list[str] | None = None,
) -> dict[str, Path]:
    """Convert all gold parquets to .hyper files. Returns {name: path} mapping."""
    hyper_dir.mkdir(parents=True, exist_ok=True)
    result = {}

    parquet_files = sorted(gold_dir.glob("*.parquet"))
    for pq_path in parquet_files:
        table_name = pq_path.stem
        if tables and table_name not in tables:
            continue

        df = pd.read_parquet(pq_path)
        hyper_path = hyper_dir / f"{table_name}.hyper"
        write_dataframe_to_hyper(df, hyper_path, table_name)
        result[table_name] = hyper_path

    return result


def export_silver_sample(
    silver_path: Path = SILVER_DIR / "transactions_enriched.parquet",
    hyper_dir: Path = HYPER_DIR,
    sample_frac: float = 0.10,
    seed: int = 42,
) -> Path:
    """Export stratified 10% Silver sample as .hyper for Tableau drill-through."""
    df = pd.read_parquet(silver_path)
    sample = df.groupby(["region", "year_sale"], group_keys=False).apply(
        lambda g: g.sample(frac=sample_frac, random_state=seed)
    )

    hyper_path = hyper_dir / "silver_sample.hyper"
    write_dataframe_to_hyper(sample, hyper_path, "silver_sample")
    return hyper_path
