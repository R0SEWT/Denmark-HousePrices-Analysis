"""
Módulo de análisis univariado
Contiene funciones para análisis univariado de variables numéricas, discretas y categóricas
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from wordcloud import WordCloud
import numpy as np
import warnings


# --- ANÁLISIS NUMÉRICO CONTINUO ---

def analyze_numeric_series(series):
    """
    Analiza una serie numérica y retorna estadísticas descriptivas.
    
    Args:
        series: Serie pandas numérica
        
    Returns:
        Diccionario con estadísticas descriptivas
    """
    summary = {
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "skew": series.skew()
    }
    try:
        stat, p = stats.normaltest(series.dropna())
        summary["normal_stat"] = stat
        summary["normal_p"] = p
        summary["is_normal"] = p >= 0.05
    except:
        summary["is_normal"] = None
    return summary


def plot_numeric_distribution(series, col):
    """
    Crea visualizaciones para una variable numérica.
    
    Args:
        series: Serie pandas numérica
        col: Nombre de la columna
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Distribución de '{col}'", fontsize=14)

    sns.histplot(series, kde=True, ax=axes[0], bins=50, color="steelblue")
    axes[0].set_title("Histograma + KDE")
    axes[0].set_xlabel(col)

    sns.boxplot(x=series, ax=axes[1], color="lightcoral")
    axes[1].set_title("Boxplot")
    axes[1].set_xlabel(col)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    stats.probplot(series.dropna(), dist="norm", plot=plt)
    plt.title(f"Q-Q plot de `{col}`")
    plt.grid()
    plt.tight_layout()
    plt.show()


def describe_numeric(df, cols):
    """
    Describe variables numéricas con estadísticas y visualizaciones.
    
    Args:
        df: DataFrame
        cols: Lista de columnas numéricas
    """
    for col in cols:
        print(f"\nAnálisis de `{col}`:")
        s = analyze_numeric_series(df[col])
        for k, v in s.items():
            print(f"  - {k}: {v:,.2f}" if isinstance(v, float) else f"  - {k}: {v}")
        plot_numeric_distribution(df[col], col)
        print("="*50)


# --- ANÁLISIS DISCRETO O CATEGÓRICO ---

def plot_discrete_distribution(series, col):
    """
    Crea visualizaciones para una variable discreta.
    
    Args:
        series: Serie pandas discreta
        col: Nombre de la columna
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Distribución de '{col}'", fontsize=14)

    sns.countplot(x=series, ax=axes[0], palette="viridis", order=series.value_counts().index)
    axes[0].set_title("Barras")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frecuencia")
    axes[0].tick_params(axis='x', rotation=45)

    series.value_counts().plot.pie(
        autopct='%1.1f%%',
        ax=axes[1],
        colors=sns.color_palette("viridis", n_colors=series.nunique())
    )
    axes[1].set_title("Pastel")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()


def describe_discrete(df, cols):
    """
    Describe variables discretas con estadísticas y visualizaciones.
    
    Args:
        df: DataFrame
        cols: Lista de columnas discretas
    """
    for col in cols:
        print(f"\nAnálisis de `{col}`:")
        print(f"  - Valores únicos: {df[col].nunique()}")
        print("  - Frecuencias:")
        print(df[col].value_counts())
        print("  - Proporciones:")
        print(df[col].value_counts(normalize=True))
        plot_discrete_distribution(df[col], col)
        print("="*50)


# --- ANÁLISIS CATEGÓRICO ---

def obtener_top_y_otros(df, col, top_n=10):
    """
    Devuelve una Serie con los top_n-1 valores más frecuentes de la columna `col`
    y agrupa el resto como 'other'. Los valores son proporciones.
    
    Args:
        df: DataFrame
        col: Nombre de la columna
        top_n: Número de categorías top a mostrar
        
    Returns:
        Serie con las proporciones
    """
    conteo = df[col].value_counts(normalize=True)
    
    if len(conteo) <= top_n:
        return conteo

    top_n_minus_1 = conteo.iloc[:top_n - 1]
    otros = conteo.iloc[top_n - 1:].sum()
    resultado = pd.concat([top_n_minus_1, pd.Series({'other': otros})])
    
    return resultado


def generar_wordcloud(df, col, max_words=100):
    """
    Genera un WordCloud para una variable categórica.
    
    Args:
        df: DataFrame
        col: Nombre de la columna
        max_words: Número máximo de palabras en el wordcloud
    """
    print(f"\n--- WordCloud para '{col}' ---")
    
    # Concatenar los valores de la columna como texto
    texto = " ".join(df[col].dropna().astype(str).values)
    
    # Crear y mostrar el WordCloud
    wc = WordCloud(width=800, height=400, background_color="white", max_words=max_words, colormap="viridis").generate(texto)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud de '{col}'", fontsize=16)
    plt.show()


def analizar_high_cardinality(df, col):
    """
    Analiza una variable categórica de alta cardinalidad.
    
    Args:
        df: DataFrame
        col: Nombre de la columna
    """
    generar_wordcloud(df, col, max_words=50)
    return


def analizar_variable_categorica(df, col, top_n=10, other_threshold=0.8):
    """
    Visualiza y analiza una variable categórica con top_n categorías como máximo (otras se agrupan).
    
    Args:
        df: DataFrame
        col: Nombre de la columna
        top_n: Número de categorías top a mostrar
        other_threshold: Umbral para considerar alta cardinalidad
    """
    print(f"\n--- Análisis de '{col}' ---")

    proporciones = obtener_top_y_otros(df, col, top_n=top_n)

    if 'other' in proporciones.index and proporciones['other'] >= other_threshold:
        analizar_high_cardinality(df, col)
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Distribución de '{col}'", fontsize=14, fontweight='bold')

    # Gráfico de barras
    sns.barplot(x=proporciones.index, y=proporciones.values, ax=axes[0], palette="viridis")
    axes[0].set_title("Top categorías")
    axes[0].set_ylabel("Proporción")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

    # Gráfico de pastel
    explode = [0.05] * (len(proporciones) - 1) + [0] if 'other' in proporciones.index else [0.05] * len(proporciones)
    proporciones.plot.pie(
        ax=axes[1],
        autopct='%1.1f%%',
        startangle=90,
        explode=explode,
        colors=sns.color_palette("viridis", n_colors=len(proporciones)),
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
        textprops={'fontsize': 10}
    )
    axes[1].set_title("Pastel")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()

    # Narrativa básica
    print(f"- Valores únicos (originales): {df[col].nunique()}")
    print(f"- Distribución mostrada:\n{proporciones.round(4)}")
    print("=" * 60)


def plot_categorical_distributions(df, categorical_cols):
    """
    Plotea las distribuciones de variables categóricas.
    
    Args:
        df: DataFrame
        categorical_cols: Lista de columnas categóricas
    """
    for col in categorical_cols:
        if col in df.columns:
            analizar_variable_categorica(df, col, top_n=5, other_threshold=0.8)
        else:
            print(f"Columna '{col}' no encontrada en el DataFrame.")


# --- FUNCIÓN PRINCIPAL ---

def run_univariate_analysis(df, continuous_cols=[], discrete_cols=[], categorical_cols=[]):
    """
    Ejecuta análisis univariado para diferentes tipos de variables.
    
    Args:
        df: DataFrame
        continuous_cols: Lista de columnas numéricas continuas
        discrete_cols: Lista de columnas discretas
        categorical_cols: Lista de columnas categóricas
    """
    if continuous_cols:
        describe_numeric(df, continuous_cols)
    if discrete_cols:
        describe_discrete(df, discrete_cols)
    if categorical_cols:
        plot_categorical_distributions(df, categorical_cols)
