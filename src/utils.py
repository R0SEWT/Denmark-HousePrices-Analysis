import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_df_null_resume_and_percentages(df: pd.DataFrame) -> pd.DataFrame:
    df_nulls = df[df.isna().any(axis=1)]
    
    df_nulls_resume = (df.isna().sum() / df.shape[0] * 100)
    df_nulls_resume = df_nulls_resume[df_nulls_resume > 0].sort_values(ascending=False).reset_index()
    df_nulls_resume.columns = ["column", "null_percentage"]
    df_nulls_resume["null_percentage"] = df_nulls_resume["null_percentage"].apply(lambda x: f"{x:.2f} %")
    return df_nulls, df_nulls_resume

def get_duplicate_percentage(df: pd.DataFrame) -> float:
    return round((df.duplicated().sum() / df.shape[0]) * 100, 2)

def plot_null_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, cmap='viridis')
    plt.title("Mapa de valores nulos")
    plt.show()

def get_column_types(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "n_missing": df.isna().sum()
    })

def plot_target_distribution(df: pd.DataFrame, target: str):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[target].dropna(), kde=True)
    plt.title(f"Distribución de {target}")
    plt.xlabel(target)
    plt.show()
