"""Layer manifests, provenance tracking, and risk documentation."""

import json
from pathlib import Path

from config import MEDALLION_METADATA_DIR


RISK_DOCUMENTATION = {
    "risks": [
        {
            "id": "R1",
            "name": "Sesgo en registros preliminares",
            "description": "1992-1995 con menor completitud en year_build y sqm",
            "impact": "Alto",
            "mitigation": "Flag is_pre_1995_record en Silver; filtrar en Gold a criterio del consumidor",
        },
        {
            "id": "R2",
            "name": "Registros con sales_type indeterminado",
            "description": "~0.5% no permite clasificar si la venta es libre o familiar",
            "impact": "Medio",
            "mitigation": "Flag is_unclassified_sale; excluir por defecto en Gold KPIs de precio",
        },
        {
            "id": "R3",
            "name": "Ventas familiares",
            "description": "Ventas entre parientes contaminan analisis de precios de mercado",
            "impact": "Medio",
            "mitigation": "Flag is_family_sale; excluir por defecto en Gold KPIs de precio",
        },
        {
            "id": "R4",
            "name": "Cobertura geografica desigual",
            "description": "Regiones rurales remotas con < 50 transacciones/anio",
            "impact": "Medio",
            "mitigation": "Flag is_low_sample_cell; n_transactions incluido en cada KPI",
        },
        {
            "id": "R5",
            "name": "Inflacion no ajustada",
            "description": "purchase_price en DKK corrientes",
            "impact": "Alto",
            "mitigation": "Silver computa real_purchase_price y real_sqm_price deflactados (base 2024)",
        },
        {
            "id": "R6",
            "name": "Tasas macro no anualizadas",
            "description": "nom_interest_rate% y dk_ann_infl_rate% son tasas anuales reportadas trimestralmente",
            "impact": "Bajo-Medio",
            "mitigation": "Documentado en metadata; CPI usa (1+r)^0.25 para compounding trimestral",
        },
        {
            "id": "R7",
            "name": "Sin coordenadas nativas",
            "description": "Solo zip_code, city, area, region — no hay geometria incorporada",
            "impact": "Medio",
            "mitigation": "Silver une contra centroides DAWA (api.dataforsyningen.dk)",
        },
        {
            "id": "R8",
            "name": "Riesgo de interpretacion causal",
            "description": "Yuxtaposicion de series de precio con tasas puede inducir lectura causal",
            "impact": "Alto",
            "mitigation": "bond_elasticity marcado como confounded_OLS; documentacion explicita",
        },
    ],
    "kpi_definitions": {
        "real_sqm_price": "purchase_price / sqm deflactado con dk_ann_infl_rate% (base 2024 DKK)",
        "price_index": "Mediana trimestral de real_sqm_price por region, base 1992=100",
        "transaction_volume": "Conteo trimestral por region y house_type",
        "drawdown": "Caida maxima porcentual desde pico previo por region",
        "volatility": "Desvio estandar rolling de 4 y 8 trimestres de mediana real_sqm_price",
        "bond_elasticity": "OLS observacional de delta%_volumen vs delta_bps_yield con rezagos 0, 1, 2 trimestres",
    },
    "temporal_conventions": {
        "granularity": "trimestre (quarter)",
        "base_year": 1992,
        "cpi_base_year": 2024,
        "rolling_windows": {"short": "4 trimestres", "long": "8 trimestres"},
        "macro_shock_markers": [1995, 2008, 2020, 2022],
    },
}


def write_risk_documentation(output_dir: Path = MEDALLION_METADATA_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "risk_documentation.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(RISK_DOCUMENTATION, f, indent=2, ensure_ascii=False)
    print(f"Risk documentation → {path}")
    return path


def validate_layer_manifests(metadata_dir: Path = MEDALLION_METADATA_DIR) -> bool:
    """Validate that all layer manifests exist and are consistent."""
    layers = ["bronze", "silver", "gold"]
    all_ok = True

    for layer in layers:
        manifest_path = metadata_dir / f"{layer}_manifest.json"
        if not manifest_path.exists():
            print(f"MISSING: {manifest_path}")
            all_ok = False
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        print(f"{layer}: {manifest.get('output_row_count', 'N/A')} rows, "
              f"written {manifest.get('written_at', 'unknown')}")

    if all_ok:
        bronze = json.loads((metadata_dir / "bronze_manifest.json").read_text())
        silver = json.loads((metadata_dir / "silver_manifest.json").read_text())
        if bronze["output_row_count"] != silver["output_row_count"]:
            print(f"WARNING: row count mismatch Bronze({bronze['output_row_count']}) "
                  f"!= Silver({silver['output_row_count']})")
            all_ok = False
        else:
            print("Row count invariant: Bronze == Silver ✓")

    return all_ok
