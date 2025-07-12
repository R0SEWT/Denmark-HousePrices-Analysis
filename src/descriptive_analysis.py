"""
Módulo de análisis descriptivo para el mercado inmobiliario danés.
Contiene todas las funciones necesarias para realizar análisis completo de KPIs regionales,
precios por m², volumen de transacciones y tendencias temporales.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h2o
from datetime import datetime
from scipy import stats
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# FUNCIONES DE UTILIDAD Y CONFIGURACIÓN
# =============================================================================

def load_and_validate_data(data_path, destination_frame='df_clean'):
    """
    Cargar y validar datos del análisis exploratorio usando H2O.
    
    Parameters:
    -----------
    data_path : str
        Ruta al archivo de datos limpio
    destination_frame : str
        Nombre del frame de destino en H2O
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con los datos cargados y validados
    """
    try:
        h2o.init()
        h2o.connect()
        print(f"Importando datos desde {data_path}\n")

        df_h2o = h2o.import_file(
            path=str(data_path),
            header=1,
            sep=",",
            destination_frame=str(destination_frame)
        )

        df_clean = df_h2o.as_data_frame()
        print(f"Datos importados a H2O con destino: {destination_frame}\n")
        print(f"Dimensiones del H2OFrame: {df_h2o.nrows:,} filas × {df_h2o.ncols} columnas\n")
        print(f"Datos cargados: {df_clean.shape[0]:,} registros x {df_clean.shape[1]} columnas")
        print(f"Período: {df_clean['date'].min()} - {df_clean['date'].max()}")
        print(f"Regiones: {df_clean['region'].nunique()}")
        print(f"Rango precios: {df_clean['purchase_price'].min():,.0f} - {df_clean['purchase_price'].max():,.0f} DKK")
        return df_clean

    except Exception as e:
        print(f"Error al cargar datos: {e}")
        raise


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calcular intervalo de confianza para una serie de datos.
    
    Parameters:
    -----------
    data : pd.Series
        Serie de datos numéricos
    confidence : float
        Nivel de confianza (por defecto 0.95 para 95%)
        
    Returns:
    --------
    tuple
        Tupla con (límite_inferior, límite_superior)
    """
    mean = data.mean()
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return mean - h, mean + h


def classify_market_size(participation):
    """
    Clasificar regiones por tamaño de mercado basado en participación.
    
    Parameters:
    -----------
    participation : float
        Porcentaje de participación en el mercado
        
    Returns:
    --------
    str
        Clasificación del mercado
    """
    if participation >= 5.0:
        return 'Principal'
    elif participation >= 2.0:
        return 'Secundario'
    elif participation >= 0.5:
        return 'Terciario'
    else:
        return 'Nicho'


def configure_plot_style():
    """Configurar estilo de visualizaciones."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


# =============================================================================
# ANÁLISIS REGIONAL DE PRECIOS
# =============================================================================

def analyze_regional_prices(df):
    """
    Análisis completo de precios por región.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de transacciones inmobiliarias
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con estadísticas regionales completas
    """
    # Estadísticas descriptivas por región
    regional_stats = df.groupby('region')['purchase_price'].agg([
        'count', 'mean', 'median', 'std',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
        'min', 'max'
    ]).round(0)
    
    regional_stats.columns = ['Transacciones', 'Promedio', 'Mediana', 'Std', 'Q1', 'Q3', 'Minimo', 'Maximo']
    
    # Calcular intervalos de confianza
    ci_data = []
    for region in df['region'].unique():
        region_prices = df[df['region'] == region]['purchase_price']
        ci_lower, ci_upper = calculate_confidence_interval(region_prices)
        ci_data.append((ci_lower, ci_upper))
    
    ci_df = pd.DataFrame(ci_data, index=regional_stats.index, 
                        columns=['CI_Lower', 'CI_Upper']).round(0)
    
    # Combinar estadísticas
    regional_stats = pd.concat([regional_stats, ci_df], axis=1)
    regional_stats = regional_stats.sort_values('Promedio', ascending=False)
    
    return regional_stats


def print_regional_summary(regional_stats, top_n=10):
    """
    Imprimir resumen de estadísticas regionales.
    
    Parameters:
    -----------
    regional_stats : pd.DataFrame
        DataFrame con estadísticas regionales
    top_n : int
        Número de regiones top a mostrar
        
    Returns:
    --------
    pd.DataFrame
        Top N regiones
    """
    print("ESTADÍSTICAS DE PRECIOS POR REGIÓN")
    print("=" * 50)
    print(f"Total regiones analizadas: {len(regional_stats)}")
    print(f"Rango precios promedio: {regional_stats['Promedio'].min():,.0f} - {regional_stats['Promedio'].max():,.0f} DKK")
    print(f"\nTOP {top_n} REGIONES MÁS CARAS")
    print("-" * 40)
    return regional_stats.head(top_n)


def create_regional_price_plots(regional_stats, df, figsize=(16, 12)):
    """
    Crear visualizaciones de precios regionales.
    
    Parameters:
    -----------
    regional_stats : pd.DataFrame
        DataFrame con estadísticas regionales
    df : pd.DataFrame
        DataFrame original con todos los datos
    figsize : tuple
        Tamaño de la figura
        
    Returns:
    --------
    pd.Series
        Serie con coeficientes de variación
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Análisis de Precios por Región en Dinamarca', fontsize=16, fontweight='bold')

    # 1. Ranking de precios promedio (Top 15)
    top_15 = regional_stats.head(15)
    bars1 = axes[0,0].barh(range(len(top_15)), top_15['Promedio'], 
                           color=sns.color_palette("viridis", len(top_15)))
    axes[0,0].set_yticks(range(len(top_15)))
    axes[0,0].set_yticklabels(top_15.index, fontsize=10)
    axes[0,0].set_xlabel('Precio Promedio (DKK)')
    axes[0,0].set_title('Top 15 Regiones por Precio Promedio')
    axes[0,0].grid(axis='x', alpha=0.3)

    # Valores en barras
    for i, v in enumerate(top_15['Promedio']):
        axes[0,0].text(v + 50000, i, f'{v:,.0f}', va='center', fontsize=9)

    # 2. Promedio vs Mediana (Top 10)
    top_10 = regional_stats.head(10)
    x = np.arange(len(top_10))
    width = 0.35

    axes[0,1].bar(x - width/2, top_10['Promedio'], width, label='Promedio', alpha=0.8)
    axes[0,1].bar(x + width/2, top_10['Mediana'], width, label='Mediana', alpha=0.8)
    axes[0,1].set_xlabel('Regiones')
    axes[0,1].set_ylabel('Precio (DKK)')
    axes[0,1].set_title('Promedio vs Mediana - Top 10')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(top_10.index, rotation=45, ha='right', fontsize=9)
    axes[0,1].legend()
    axes[0,1].grid(axis='y', alpha=0.3)

    # 3. Distribución de precios promedio
    axes[1,0].hist(regional_stats['Promedio'], bins=20, alpha=0.7, edgecolor='black')
    mean_price = regional_stats['Promedio'].mean()
    median_price = regional_stats['Promedio'].median()
    
    axes[1,0].axvline(mean_price, color='red', linestyle='--', 
                      label=f'Media: {mean_price:,.0f} DKK')
    axes[1,0].axvline(median_price, color='orange', linestyle='--',
                      label=f'Mediana: {median_price:,.0f} DKK')
    axes[1,0].set_xlabel('Precio Promedio (DKK)')
    axes[1,0].set_ylabel('Número de Regiones')
    axes[1,0].set_title('Distribución de Precios Promedio')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)

    # 4. Coeficiente de variación
    cv_data = ((regional_stats['Std'] / regional_stats['Promedio']) * 100).sort_values(ascending=False).head(15)
    axes[1,1].bar(range(len(cv_data)), cv_data, alpha=0.8)
    axes[1,1].set_xticks(range(len(cv_data)))
    axes[1,1].set_xticklabels(cv_data.index, rotation=45, ha='right', fontsize=9)
    axes[1,1].set_ylabel('Coeficiente de Variación (%)')
    axes[1,1].set_title('Variabilidad por Región (Top 15)')
    axes[1,1].grid(axis='y', alpha=0.3)

    # Valores en barras
    for i, v in enumerate(cv_data):
        axes[1,1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
    
    return cv_data


def print_regional_insights(regional_stats, cv_data):
    """
    Imprimir insights clave del análisis regional.
    
    Parameters:
    -----------
    regional_stats : pd.DataFrame
        DataFrame con estadísticas regionales
    cv_data : pd.Series
        Serie con coeficientes de variación
    """
    print("\nINSIGHTS CLAVE - PRECIOS REGIONALES")
    print("=" * 40)
    print(f"Región más cara: {regional_stats.index[0]} ({regional_stats.iloc[0]['Promedio']:,.0f} DKK)")
    print(f"Región más económica: {regional_stats.index[-1]} ({regional_stats.iloc[-1]['Promedio']:,.0f} DKK)")
    print(f"Ratio precio max/min: {regional_stats.iloc[0]['Promedio'] / regional_stats.iloc[-1]['Promedio']:.1f}x")
    print(f"Regiones sobre la media: {(regional_stats['Promedio'] > regional_stats['Promedio'].mean()).sum()}")
    print(f"CV promedio: {cv_data.mean():.1f}%")


# =============================================================================
# ANÁLISIS DE PRECIO POR M²
# =============================================================================

def analyze_sqm_prices(df):
    """
    Análisis de precio por m² por región.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de transacciones
        
    Returns:
    --------
    tuple
        (sqm_stats, premium_threshold)
    """
    # Estadísticas básicas
    sqm_stats = df.groupby('region')['sqm_price'].agg([
        'count', 'mean', 'median', 'std',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
        'min', 'max'
    ]).round(0)
    
    sqm_stats.columns = ['Transacciones', 'Promedio_m2', 'Mediana_m2', 'Std_m2', 
                        'Q1_m2', 'Q3_m2', 'Min_m2', 'Max_m2']
    sqm_stats = sqm_stats.sort_values('Promedio_m2', ascending=False)
    
    # Identificar mercados premium
    premium_threshold = df['sqm_price'].quantile(0.75)
    sqm_stats['Es_Premium'] = sqm_stats['Promedio_m2'] > premium_threshold
    
    # Coeficiente de variación
    sqm_stats['CV_m2'] = (sqm_stats['Std_m2'] / sqm_stats['Promedio_m2']) * 100
    
    return sqm_stats, premium_threshold


def create_ranking_comparison(regional_stats, sqm_stats):
    """
    Crear comparación de rankings entre precio total y precio/m².
    
    Parameters:
    -----------
    regional_stats : pd.DataFrame
        Estadísticas regionales de precio total
    sqm_stats : pd.DataFrame
        Estadísticas regionales de precio por m²
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con comparación de rankings
    """
    comparison = pd.DataFrame({
        'Region': regional_stats.index,
        'Precio_Total': regional_stats['Promedio'].values,
        'Precio_m2': sqm_stats.loc[regional_stats.index, 'Promedio_m2'].values,
        'Rank_Total': range(1, len(regional_stats) + 1),
        'Rank_m2': range(1, len(sqm_stats) + 1)
    })
    comparison['Diferencia_Rank'] = comparison['Rank_Total'] - comparison['Rank_m2']
    return comparison


def create_sqm_price_plots(df, sqm_stats, premium_threshold, comparison_df, figsize=(16, 12)):
    """
    Crear visualizaciones de análisis de precio por m².
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame original
    sqm_stats : pd.DataFrame
        Estadísticas de precio por m²
    premium_threshold : float
        Umbral para mercados premium
    comparison_df : pd.DataFrame
        DataFrame con comparación de rankings
    figsize : tuple
        Tamaño de la figura
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Análisis de Precio por m² por Región', fontsize=16, fontweight='bold')

    # 1. Top 15 regiones por precio/m²
    top_15_sqm = sqm_stats.head(15)
    colors = ['red' if premium else 'blue' for premium in top_15_sqm['Es_Premium']]
    
    bars1 = ax1.bar(range(len(top_15_sqm)), top_15_sqm['Promedio_m2'], color=colors, alpha=0.7)
    ax1.set_title('Top 15 Regiones - Precio por m²')
    ax1.set_ylabel('Precio por m² (DKK)')
    ax1.set_xlabel('Región')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.set_xticks(range(len(top_15_sqm)))
    ax1.set_xticklabels(top_15_sqm.index, ha='right')
    ax1.axhline(y=premium_threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Umbral Premium ({premium_threshold:,.0f})')
    ax1.legend()

    # 2. Comparación normalizada
    top_10_comp = comparison_df.head(10)
    x_pos = np.arange(len(top_10_comp))
    width = 0.35
    
    # Normalizar para comparación visual
    precio_total_norm = (top_10_comp['Precio_Total'] / top_10_comp['Precio_Total'].max()) * 100
    precio_m2_norm = (top_10_comp['Precio_m2'] / top_10_comp['Precio_m2'].max()) * 100
    
    ax2.bar(x_pos - width/2, precio_total_norm, width, label='Precio Total', alpha=0.8)
    ax2.bar(x_pos + width/2, precio_m2_norm, width, label='Precio/m²', alpha=0.8)
    ax2.set_title('Comparación Normalizada: Precio Total vs Precio/m²')
    ax2.set_ylabel('Valor Normalizado (%)')
    ax2.set_xlabel('Región (Top 10)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(top_10_comp['Region'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Distribución nacional de precio/m²
    ax3.hist(df['sqm_price'], bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(df['sqm_price'].mean(), color='red', linestyle='-', linewidth=2, 
               label=f'Media: {df["sqm_price"].mean():,.0f}')
    ax3.axvline(df['sqm_price'].median(), color='green', linestyle='--', linewidth=2,
               label=f'Mediana: {df["sqm_price"].median():,.0f}')
    ax3.axvline(premium_threshold, color='orange', linestyle=':', linewidth=2,
               label=f'Umbral Premium: {premium_threshold:,.0f}')
    ax3.set_title('Distribución Nacional de Precio por m²')
    ax3.set_xlabel('Precio por m² (DKK)')
    ax3.set_ylabel('Frecuencia')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Variabilidad por región
    top_cv = sqm_stats.sort_values('CV_m2', ascending=False).head(15)
    ax4.bar(range(len(top_cv)), top_cv['CV_m2'], alpha=0.8)
    ax4.set_title('Variabilidad del Precio/m² por Región')
    ax4.set_ylabel('Coeficiente de Variación (%)')
    ax4.set_xlabel('Región')
    ax4.set_xticks(range(len(top_cv)))
    ax4.set_xticklabels(top_cv.index, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    # Valores en barras
    for i, v in enumerate(top_cv['CV_m2']):
        ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def print_sqm_insights(sqm_stats, comparison_df):
    """
    Imprimir insights del análisis de precio por m².
    
    Parameters:
    -----------
    sqm_stats : pd.DataFrame
        Estadísticas de precio por m²
    comparison_df : pd.DataFrame
        Comparación de rankings
    """
    print("\nINSIGHTS CLAVE - PRECIO POR M²")
    print("=" * 35)
    print(f"Región más eficiente: {sqm_stats.index[0]}")
    print(f"Precio/m² máximo: {sqm_stats.iloc[0]['Promedio_m2']:,.0f} DKK/m²")
    top_cv = sqm_stats.sort_values('CV_m2', ascending=False)
    print(f"Región más variable: {top_cv.index[0]} ({top_cv.iloc[0]['CV_m2']:.1f}%)")
    print(f"Región más estable: {sqm_stats.sort_values('CV_m2').index[0]} ({sqm_stats.sort_values('CV_m2').iloc[0]['CV_m2']:.1f}%)")
    print(f"Mayor diferencia ranking: {abs(comparison_df['Diferencia_Rank']).max()} posiciones")


# =============================================================================
# ANÁLISIS DE VOLUMEN DE TRANSACCIONES
# =============================================================================

def analyze_transaction_volume(df):
    """
    Análisis de volumen de transacciones por región.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de transacciones
        
    Returns:
    --------
    tuple
        (volume_stats, correlation, high_liquidity_threshold)
    """
    # Estadísticas de volumen
    volume_stats = df.groupby('region').agg({
        'purchase_price': ['count', 'sum'],
        'sqm_price': 'mean'
    }).round(0)
    
    volume_stats.columns = ['Num_Transacciones', 'Volumen_Total_DKK', 'Precio_Promedio_m2']
    volume_stats = volume_stats.sort_values('Num_Transacciones', ascending=False)
    
    # Participación de mercado
    total_transactions = volume_stats['Num_Transacciones'].sum()
    volume_stats['Participacion_Mercado'] = (volume_stats['Num_Transacciones'] / total_transactions) * 100
    volume_stats['Participacion_Acumulada'] = volume_stats['Participacion_Mercado'].cumsum()
    
    # Clasificación por tamaño de mercado
    volume_stats['Tipo_Mercado'] = volume_stats['Participacion_Mercado'].apply(classify_market_size)
    
    # Correlación volumen-precio
    correlation = np.corrcoef(volume_stats['Num_Transacciones'], volume_stats['Precio_Promedio_m2'])[0,1]
    
    # Alta liquidez (top 20%)
    high_liquidity_threshold = volume_stats['Num_Transacciones'].quantile(0.8)
    volume_stats['Alta_Liquidez'] = volume_stats['Num_Transacciones'] > high_liquidity_threshold
    
    return volume_stats, correlation, high_liquidity_threshold


def print_volume_summary(volume_stats, correlation, high_liquidity_threshold):
    """
    Imprimir resumen del análisis de volumen.
    
    Parameters:
    -----------
    volume_stats : pd.DataFrame
        Estadísticas de volumen
    correlation : float
        Correlación volumen-precio
    high_liquidity_threshold : float
        Umbral de alta liquidez
    """
    print("ANÁLISIS DE VOLUMEN DE TRANSACCIONES")
    print("=" * 45)
    print(f"Total transacciones: {volume_stats['Num_Transacciones'].sum():,}")
    print(f"Volumen total: {volume_stats['Volumen_Total_DKK'].sum():,.0f} DKK")
    print(f"Correlación volumen-precio/m²: {correlation:.3f}")
    print(f"Regiones alta liquidez: {volume_stats['Alta_Liquidez'].sum()}")
    
    # Análisis de concentración
    pareto_80 = volume_stats[volume_stats['Participacion_Acumulada'] <= 80]
    print(f"\nCONCENTRACIÓN DE MERCADO")
    print("-" * 25)
    print(f"Regiones que concentran 80% del mercado: {len(pareto_80)}")
    print(f"Participación top 10: {volume_stats.head(10)['Participacion_Mercado'].sum():.1f}%")
    
    # Distribución por tipo
    market_dist = volume_stats['Tipo_Mercado'].value_counts()
    print(f"\nDISTRIBUCIÓN POR TIPO DE MERCADO")
    print("-" * 30)
    for market_type, count in market_dist.items():
        percentage = (count / len(volume_stats)) * 100
        print(f"{market_type}: {count} regiones ({percentage:.1f}%)")


def create_volume_plots(volume_stats, correlation, high_liquidity_threshold, figsize=(16, 12)):
    """
    Crear visualizaciones de análisis de volumen.
    
    Parameters:
    -----------
    volume_stats : pd.DataFrame
        Estadísticas de volumen
    correlation : float
        Correlación volumen-precio
    high_liquidity_threshold : float
        Umbral de alta liquidez
    figsize : tuple
        Tamaño de la figura
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Análisis de Volumen de Transacciones por Región', fontsize=16, fontweight='bold')

    # Mapeo de colores para tipos de mercado
    market_colors = {'Principal': 'red', 'Secundario': 'blue', 'Terciario': 'green', 'Nicho': 'gray'}

    # 1. Top 20 regiones por volumen
    top_20 = volume_stats.head(20)
    colors = [market_colors[market] for market in top_20['Tipo_Mercado']]
    
    bars1 = ax1.bar(range(len(top_20)), top_20['Num_Transacciones'], color=colors, alpha=0.7)
    ax1.set_title('Top 20 Regiones por Volumen de Transacciones')
    ax1.set_ylabel('Número de Transacciones')
    ax1.set_xlabel('Región')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.set_xticks(range(len(top_20)))
    ax1.set_xticklabels(top_20.index, ha='right')
    ax1.axhline(y=high_liquidity_threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Umbral Alta Liquidez ({high_liquidity_threshold:,.0f})')
    ax1.legend()

    # 2. Diagrama de Pareto
    x_pos = range(len(top_20))
    ax2_twin = ax2.twinx()
    
    # Barras de participación individual
    ax2.bar(x_pos, top_20['Participacion_Mercado'], alpha=0.7, label='Participación Individual')
    # Línea de participación acumulada
    ax2_twin.plot(x_pos, top_20['Participacion_Acumulada'], color='red', marker='o', 
                  linewidth=2, label='Participación Acumulada')
    
    ax2.set_title('Diagrama de Pareto - Concentración del Mercado')
    ax2.set_ylabel('Participación Individual (%)', color='blue')
    ax2_twin.set_ylabel('Participación Acumulada (%)', color='red')
    ax2.set_xlabel('Región (Top 20)')
    ax2.set_xticks(x_pos[::2])
    ax2.set_xticklabels(top_20.index[::2], rotation=45, ha='right')
    ax2_twin.axhline(y=80, color='green', linestyle=':', alpha=0.7, label='Regla 80/20')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    # 3. Relación Volumen vs Precio/m²
    scatter_colors = [market_colors[market] for market in volume_stats['Tipo_Mercado']]
    ax3.scatter(volume_stats['Num_Transacciones'], volume_stats['Precio_Promedio_m2'], 
               c=scatter_colors, alpha=0.7, s=50)
    ax3.set_xscale('log')
    ax3.set_title(f'Relación Volumen vs Precio/m² (r={correlation:.3f})')
    ax3.set_xlabel('Número de Transacciones (escala log)')
    ax3.set_ylabel('Precio Promedio por m² (DKK)')
    ax3.grid(True, alpha=0.3)
    
    # Línea de tendencia
    log_volume = np.log10(volume_stats['Num_Transacciones'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_volume, volume_stats['Precio_Promedio_m2'])
    line_x = np.linspace(volume_stats['Num_Transacciones'].min(), volume_stats['Num_Transacciones'].max(), 100)
    line_y = slope * np.log10(line_x) + intercept
    ax3.plot(line_x, line_y, 'r--', alpha=0.8, label=f'Tendencia (R²={r_value**2:.3f})')
    ax3.legend()

    # 4. Distribución de tipos de mercado
    market_counts = volume_stats['Tipo_Mercado'].value_counts()
    colors_pie = [market_colors[market] for market in market_counts.index]
    
    wedges, texts, autotexts = ax4.pie(market_counts.values, labels=market_counts.index, 
                                      colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Distribución de Regiones por Tipo de Mercado')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    plt.tight_layout()
    plt.show()


def print_volume_insights(volume_stats, correlation):
    """
    Imprimir insights del análisis de volumen.
    
    Parameters:
    -----------
    volume_stats : pd.DataFrame
        Estadísticas de volumen
    correlation : float
        Correlación volumen-precio
    """
    print("\nINSIGHTS CLAVE - VOLUMEN DE TRANSACCIONES")
    print("=" * 45)
    print(f"Región líder: {volume_stats.index[0]} ({volume_stats.iloc[0]['Num_Transacciones']:,} trans.)")
    print(f"Participación del líder: {volume_stats.iloc[0]['Participacion_Mercado']:.1f}%")
    print(f"Mercados principales: {(volume_stats['Tipo_Mercado'] == 'Principal').sum()} regiones")
    print(f"Concentración top 5: {volume_stats.head(5)['Participacion_Mercado'].sum():.1f}%")
    print(f"Correlación volumen-precio: {'Positiva' if correlation > 0 else 'Negativa'} ({abs(correlation):.3f})")


# =============================================================================
# ANÁLISIS TEMPORAL
# =============================================================================

def analyze_temporal_trends(df):
    """
    Análisis de tendencias temporales de precios.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de transacciones
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con estadísticas anuales
    """
    # Crear columnas de fecha
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    
    # Estadísticas anuales
    yearly_stats = df.groupby('year').agg({
        'purchase_price': ['count', 'mean', 'median', 'std'],
        'sqm_price': ['mean', 'median']
    }).round(0)
    
    yearly_stats.columns = ['Transacciones', 'Precio_Promedio', 'Precio_Mediana', 'Precio_Std',
                           'Precio_m2_Promedio', 'Precio_m2_Mediana']
    
    # Calcular tasas de crecimiento anual
    yearly_stats['Crecimiento_Precio'] = yearly_stats['Precio_Promedio'].pct_change() * 100
    yearly_stats['Crecimiento_m2'] = yearly_stats['Precio_m2_Promedio'].pct_change() * 100
    
    return yearly_stats


def create_temporal_plots(yearly_stats, figsize=(16, 10)):
    """
    Crear visualizaciones de tendencias temporales.
    
    Parameters:
    -----------
    yearly_stats : pd.DataFrame
        Estadísticas anuales
    figsize : tuple
        Tamaño de la figura
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Evolución Temporal del Mercado Inmobiliario Danés', fontsize=16, fontweight='bold')
    
    years = yearly_stats.index
    
    # 1. Evolución de precios promedio
    ax1.plot(years, yearly_stats['Precio_Promedio'], marker='o', linewidth=2, markersize=4)
    ax1.set_title('Evolución del Precio Promedio')
    ax1.set_xlabel('Año')
    ax1.set_ylabel('Precio Promedio (DKK)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Evolución de precio por m²
    ax2.plot(years, yearly_stats['Precio_m2_Promedio'], marker='s', linewidth=2, 
             markersize=4, color='orange')
    ax2.set_title('Evolución del Precio por m²')
    ax2.set_xlabel('Año')
    ax2.set_ylabel('Precio por m² (DKK)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Volumen de transacciones
    ax3.bar(years, yearly_stats['Transacciones'], alpha=0.7, color='green')
    ax3.set_title('Volumen de Transacciones por Año')
    ax3.set_xlabel('Año')
    ax3.set_ylabel('Número de Transacciones')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Tasas de crecimiento
    ax4.plot(years[1:], yearly_stats['Crecimiento_Precio'].dropna(), marker='o', 
             linewidth=2, label='Precio Total', alpha=0.8)
    ax4.plot(years[1:], yearly_stats['Crecimiento_m2'].dropna(), marker='s', 
             linewidth=2, label='Precio/m²', alpha=0.8)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.set_title('Tasas de Crecimiento Anual')
    ax4.set_xlabel('Año')
    ax4.set_ylabel('Crecimiento (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def print_temporal_insights(yearly_stats):
    """
    Imprimir insights del análisis temporal.
    
    Parameters:
    -----------
    yearly_stats : pd.DataFrame
        Estadísticas anuales
    """
    print("INSIGHTS TEMPORALES")
    print("=" * 25)
    
    # Períodos de mayor crecimiento
    max_growth_year = yearly_stats['Crecimiento_Precio'].idxmax()
    max_growth_rate = yearly_stats['Crecimiento_Precio'].max()
    
    # Períodos de mayor declive
    min_growth_year = yearly_stats['Crecimiento_Precio'].idxmin()
    min_growth_rate = yearly_stats['Crecimiento_Precio'].min()
    
    print(f"Mayor crecimiento: {max_growth_year} ({max_growth_rate:.1f}%)")
    print(f"Mayor declive: {min_growth_year} ({min_growth_rate:.1f}%)")
    print(f"Crecimiento promedio anual: {yearly_stats['Crecimiento_Precio'].mean():.1f}%")
    print(f"Período analizado: {yearly_stats.index.min()} - {yearly_stats.index.max()}")


# =============================================================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS COMPLETO
# =============================================================================

def run_complete_descriptive_analysis(df):
    """
    Función principal para ejecutar análisis descriptivo completo.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de transacciones inmobiliarias
        
    Returns:
    --------
    dict
        Diccionario con todos los resultados del análisis
    """
    print("=" * 60)
    print("ANÁLISIS DESCRIPTIVO COMPLETO - MERCADO INMOBILIARIO DANÉS")
    print("=" * 60)
    
    # Configurar estilo de plots
    configure_plot_style()
    
    # 1. Análisis Regional
    print("\n1. ANÁLISIS REGIONAL")
    print("=" * 20)
    
    regional_stats = analyze_regional_prices(df)
    top_regions = print_regional_summary(regional_stats)
    cv_data = create_regional_price_plots(regional_stats, df)
    print_regional_insights(regional_stats, cv_data)
    
    # 2. Análisis de Precio por m²
    print("\n\n2. ANÁLISIS DE PRECIO POR M²")
    print("=" * 30)
    
    sqm_stats, premium_threshold = analyze_sqm_prices(df)
    comparison_df = create_ranking_comparison(regional_stats, sqm_stats)
    create_sqm_price_plots(df, sqm_stats, premium_threshold, comparison_df)
    print_sqm_insights(sqm_stats, comparison_df)
    
    # 3. Análisis de Volumen
    print("\n\n3. ANÁLISIS DE VOLUMEN DE TRANSACCIONES")
    print("=" * 40)
    
    volume_stats, correlation, high_liquidity_threshold = analyze_transaction_volume(df)
    print_volume_summary(volume_stats, correlation, high_liquidity_threshold)
    create_volume_plots(volume_stats, correlation, high_liquidity_threshold)
    print_volume_insights(volume_stats, correlation)
    
    # 4. Análisis Temporal (si está disponible)
    yearly_stats = None
    if 'date' in df.columns:
        print("\n\n4. ANÁLISIS TEMPORAL")
        print("=" * 20)
        
        yearly_stats = analyze_temporal_trends(df)
        create_temporal_plots(yearly_stats)
        print_temporal_insights(yearly_stats)
    
    # Resumen ejecutivo
    print("\n\n" + "=" * 60)
    print("RESUMEN EJECUTIVO")
    print("=" * 60)
    
    print(f"Dataset analizado: {len(df):,} transacciones")
    print(f"Regiones analizadas: {df['region'].nunique()}")
    print(f"Precio promedio nacional: {df['purchase_price'].mean():,.0f} DKK")
    print(f"Precio/m² promedio nacional: {df['sqm_price'].mean():,.0f} DKK/m²")
    print(f"Región más cara: {regional_stats.index[0]}")
    print(f"Región con mayor volumen: {volume_stats.index[0]}")
    
    return {
        'regional_stats': regional_stats,
        'sqm_stats': sqm_stats,
        'volume_stats': volume_stats,
        'yearly_stats': yearly_stats,
        'comparison_df': comparison_df,
        'premium_threshold': premium_threshold,
        'correlation': correlation,
        'high_liquidity_threshold': high_liquidity_threshold,
        'cv_data': cv_data
    }


# =============================================================================
# FUNCIONES AUXILIARES PARA EXPORTACIÓN DE RESULTADOS
# =============================================================================

def export_results_to_csv(results, output_dir='results/tablas/'):
    """
    Exportar resultados del análisis a archivos CSV.
    
    Parameters:
    -----------
    results : dict
        Diccionario con resultados del análisis
    output_dir : str
        Directorio de salida para los archivos
    """
    import os
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Exportar cada resultado
    for key, value in results.items():
        if isinstance(value, pd.DataFrame) and value is not None:
            filename = f"{output_dir}descriptive_analysis_{key}.csv"
            value.to_csv(filename)
            print(f"Exportado: {filename}")


def generate_summary_report(results):
    """
    Generar reporte resumen del análisis descriptivo.
    
    Parameters:
    -----------
    results : dict
        Diccionario con resultados del análisis
        
    Returns:
    --------
    str
        Reporte en formato texto
    """
    report = []
    report.append("=" * 80)
    report.append("REPORTE RESUMEN - ANÁLISIS DESCRIPTIVO")
    report.append("=" * 80)
    
    if 'regional_stats' in results:
        regional = results['regional_stats']
        report.append(f"\n1. ANÁLISIS REGIONAL:")
        report.append(f"   • Total regiones: {len(regional)}")
        report.append(f"   • Región más cara: {regional.index[0]}")
        report.append(f"   • Precio promedio máximo: {regional.iloc[0]['Promedio']:,.0f} DKK")
    
    if 'sqm_stats' in results:
        sqm = results['sqm_stats']
        report.append(f"\n2. ANÁLISIS PRECIO/M²:")
        report.append(f"   • Región más eficiente: {sqm.index[0]}")
        report.append(f"   • Precio/m² máximo: {sqm.iloc[0]['Promedio_m2']:,.0f} DKK/m²")
        report.append(f"   • Regiones premium: {sqm['Es_Premium'].sum()}")
    
    if 'volume_stats' in results:
        volume = results['volume_stats']
        report.append(f"\n3. ANÁLISIS VOLUMEN:")
        report.append(f"   • Región líder: {volume.index[0]}")
        report.append(f"   • Total transacciones: {volume['Num_Transacciones'].sum():,}")
        report.append(f"   • Concentración top 10: {volume.head(10)['Participacion_Mercado'].sum():.1f}%")
    
    return "\n".join(report)

# 8. Análisis por Tipo de Propiedad
def analyze_property_types(df):
    """
    Análisis completo por tipo de propiedad.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset limpio con precios de vivienda
        
    Returns:
    --------
    dict : Diccionario con todos los análisis por tipo
    """
    print("=== ANÁLISIS POR TIPO DE PROPIEDAD ===")
    
    results = {}
    
    # 8.1 Estadísticas básicas por tipo
    print("\n8.1 Estadísticas de precios por tipo de propiedad:")
    price_stats = df.groupby('house_type')[TARGET].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(0)
    price_stats['cv'] = (price_stats['std'] / price_stats['mean'] * 100).round(2)
    price_stats = price_stats.sort_values('median', ascending=False)
    
    display(price_stats.style.format({
        'count': '{:,.0f}',
        'mean': '{:,.0f} DKK',
        'median': '{:,.0f} DKK', 
        'std': '{:,.0f} DKK',
        'min': '{:,.0f} DKK',
        'max': '{:,.0f} DKK',
        'cv': '{:.1f}%'
    }))
    results['price_stats'] = price_stats
    
    # 8.2 Características físicas por tipo
    print("\n8.2 Características físicas por tipo:")
    physical_stats = df.groupby('house_type').agg({
        'sqm': ['mean', 'median', 'std'],
        'no_rooms': ['mean', 'median', 'std'],
        'sqm_price': ['mean', 'median', 'std']
    }).round(2)
    
    # Aplanar columnas multinivel
    physical_stats.columns = ['_'.join(col).strip() for col in physical_stats.columns]
    display(physical_stats.style.format('{:.1f}'))
    results['physical_stats'] = physical_stats
    
    # 8.3 Distribución regional por tipo
    print("\n8.3 Distribución regional por tipo de propiedad:")
    regional_dist = pd.crosstab(df['region'], df['house_type'], normalize='columns') * 100
    regional_dist = regional_dist.round(1)
    print("Top 5 regiones por concentración de cada tipo:")
    for house_type in regional_dist.columns:
        top_regions = regional_dist[house_type].nlargest(5)
        print(f"\n{house_type}:")
        for region, pct in top_regions.items():
            print(f"  {region}: {pct:.1f}%")
    results['regional_distribution'] = regional_dist
    
    return results

def analyze_market_behavior(df):
    """
    Análisis del comportamiento de mercado.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset limpio con precios de vivienda
        
    Returns:
    --------
    dict : Diccionario con análisis de mercado
    """
    print("=== ANÁLISIS DEL COMPORTAMIENTO DE MERCADO ===")
    
    results = {}
    
    # 4.1 Análisis por tipo de venta
    print("\n4.1 Análisis por tipo de venta (sales_type):")
    if 'sales_type' in df.columns:
        sales_analysis = df.groupby('sales_type').agg({
            TARGET: ['count', 'mean', 'median', 'std'],
            'sqm_price': ['mean', 'median']
        }).round(0)
        sales_analysis.columns = ['_'.join(col) for col in sales_analysis.columns]
        display(sales_analysis.style.format({
            f'{TARGET}_count': '{:,.0f}',
            f'{TARGET}_mean': '{:,.0f} DKK',
            f'{TARGET}_median': '{:,.0f} DKK',
            f'{TARGET}_std': '{:,.0f} DKK',
            'sqm_price_mean': '{:,.0f} DKK/m²',
            'sqm_price_median': '{:,.0f} DKK/m²'
        }))
        results['sales_type_analysis'] = sales_analysis
    
    # 4.2 Análisis de cambio entre oferta y compra
    print("\n4.2 Análisis de variación precio oferta vs compra:")
    if '%_change_between_offer_and_purchase' in df.columns:
        change_stats = df['%_change_between_offer_and_purchase'].describe()
        print(f"Estadísticas de cambio oferta-compra:")
        print(f"Media: {change_stats['mean']:.2f}%")
        print(f"Mediana: {change_stats['50%']:.2f}%")
        print(f"Desv. Std: {change_stats['std']:.2f}%")
        print(f"Rango: {change_stats['min']:.2f}% a {change_stats['max']:.2f}%")
        
        # Categorizar cambios
        df_temp = df.copy()
        df_temp['change_category'] = pd.cut(
            df_temp['%_change_between_offer_and_purchase'],
            bins=[-float('inf'), -5, -1, 1, 5, float('inf')],
            labels=['Descuento >5%', 'Descuento 1-5%', 'Sin cambio ±1%', 'Premium 1-5%', 'Premium >5%']
        )
        change_dist = df_temp['change_category'].value_counts()
        print(f"\nDistribución de cambios:")
        for cat, count in change_dist.items():
            pct = count / len(df_temp) * 100
            print(f"  {cat}: {count:,} ({pct:.1f}%)")
        
        results['price_change_analysis'] = {
            'stats': change_stats,
            'distribution': change_dist
        }
    
    # 4.3 Análisis temporal por trimestre
    print("\n4.3 Análisis por trimestre:")
    if 'quarter' in df.columns:
        quarterly_stats = df.groupby('quarter').agg({
            TARGET: ['count', 'mean', 'median'],
            'sqm_price': ['mean', 'median']
        }).round(0)
        quarterly_stats.columns = ['_'.join(col) for col in quarterly_stats.columns]
        display(quarterly_stats.style.format({
            f'{TARGET}_count': '{:,.0f}',
            f'{TARGET}_mean': '{:,.0f} DKK',
            f'{TARGET}_median': '{:,.0f} DKK',
            'sqm_price_mean': '{:,.0f} DKK/m²',
            'sqm_price_median': '{:,.0f} DKK/m²'
        }))
        results['quarterly_analysis'] = quarterly_stats
    
    return results

def analyze_market_segmentation(df):
    """
    Análisis de segmentación de mercado.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset limpio con precios de vivienda
        
    Returns:
    --------
    dict : Diccionario con análisis de segmentación
    """
    print("=== SEGMENTACIÓN DE MERCADO ===")
    
    results = {}
    
    # 5.1 Segmentación por precio (premium vs económico)
    print("\n5.1 Segmentación premium vs económico:")
    
    # Definir umbrales
    q25 = df[TARGET].quantile(0.25)
    q75 = df[TARGET].quantile(0.75) 
    q90 = df[TARGET].quantile(0.90)
    
    df_temp = df.copy()
    df_temp['price_segment'] = pd.cut(
        df_temp[TARGET],
        bins=[0, q25, q75, q90, float('inf')],
        labels=['Económico', 'Medio', 'Alto', 'Premium']
    )
    
    segment_stats = df_temp.groupby('price_segment').agg({
        TARGET: ['count', 'mean', 'median', 'min', 'max'],
        'sqm': ['mean', 'median'],
        'no_rooms': ['mean', 'median'],
        'sqm_price': ['mean', 'median']
    }).round(0)
    
    segment_stats.columns = ['_'.join(col) for col in segment_stats.columns]
    display(segment_stats.style.format({
        f'{TARGET}_count': '{:,.0f}',
        f'{TARGET}_mean': '{:,.0f} DKK',
        f'{TARGET}_median': '{:,.0f} DKK',
        f'{TARGET}_min': '{:,.0f} DKK',
        f'{TARGET}_max': '{:,.0f} DKK',
        'sqm_mean': '{:.0f} m²',
        'sqm_median': '{:.0f} m²',
        'no_rooms_mean': '{:.1f}',
        'no_rooms_median': '{:.1f}',
        'sqm_price_mean': '{:,.0f} DKK/m²',
        'sqm_price_median': '{:,.0f} DKK/m²'
    }))
    results['price_segmentation'] = segment_stats
    
    # 5.2 Segmentación por antigüedad
    print("\n5.2 Segmentación por antigüedad de la propiedad:")
    if 'year_build' in df.columns:
        current_year = 2024
        df_temp['property_age'] = current_year - df_temp['year_build']
        df_temp['age_category'] = pd.cut(
            df_temp['property_age'],
            bins=[0, 10, 25, 50, 100, float('inf')],
            labels=['Nueva (0-10 años)', 'Moderna (11-25 años)', 
                   'Madura (26-50 años)', 'Antigua (51-100 años)', 'Histórica (>100 años)']
        )
        
        age_stats = df_temp.groupby('age_category').agg({
            TARGET: ['count', 'mean', 'median'],
            'sqm_price': ['mean', 'median'],
            'property_age': ['mean', 'median']
        }).round(0)
        
        age_stats.columns = ['_'.join(col) for col in age_stats.columns]
        display(age_stats.style.format({
            f'{TARGET}_count': '{:,.0f}',
            f'{TARGET}_mean': '{:,.0f} DKK',
            f'{TARGET}_median': '{:,.0f} DKK',
            'sqm_price_mean': '{:,.0f} DKK/m²',
            'sqm_price_median': '{:,.0f} DKK/m²',
            'property_age_mean': '{:.0f} años',
            'property_age_median': '{:.0f} años'
        }))
        results['age_segmentation'] = age_stats
    
    # 5.3 Análisis de nichos especializados
    print("\n5.3 Análisis de mercados de nicho:")
    niche_types = ['Farm', 'Summerhouse']
    available_niches = [nt for nt in niche_types if nt in df['house_type'].values]
    
    if available_niches:
        niche_analysis = {}
        for niche in available_niches:
            niche_data = df[df['house_type'] == niche]
            
            print(f"\n--- Análisis de {niche} ---")
            print(f"Número de propiedades: {len(niche_data):,}")
            print(f"Precio promedio: {niche_data[TARGET].mean():,.0f} DKK")
            print(f"Precio mediano: {niche_data[TARGET].median():,.0f} DKK")
            print(f"Tamaño promedio: {niche_data['sqm'].mean():.0f} m²")
            print(f"Precio/m² promedio: {niche_data['sqm_price'].mean():,.0f} DKK/m²")
            
            # Top regiones para este nicho
            top_regions = niche_data['region'].value_counts().head(5)
            print(f"Top 5 regiones:")
            for region, count in top_regions.items():
                pct = count / len(niche_data) * 100
                print(f"  {region}: {count} ({pct:.1f}%)")
            
            niche_analysis[niche] = {
                'count': len(niche_data),
                'price_stats': niche_data[TARGET].describe(),
                'top_regions': top_regions
            }
        
        results['niche_analysis'] = niche_analysis
    
    return results

def visualize_property_type_analysis(df, property_analysis):
    """
    Visualizaciones para análisis por tipo de propiedad.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset limpio
    property_analysis : dict
        Resultados del análisis por tipo de propiedad
    """
    configure_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis por Tipo de Propiedad', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Boxplot de precios por tipo
    df.boxplot(column=TARGET, by='house_type', ax=axes[0,0])
    axes[0,0].set_title('Distribución de Precios por Tipo de Propiedad')
    axes[0,0].set_xlabel('Tipo de Propiedad')
    axes[0,0].set_ylabel('Precio (DKK)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Precio por m² por tipo
    df.boxplot(column='sqm_price', by='house_type', ax=axes[0,1])
    axes[0,1].set_title('Precio por m² por Tipo de Propiedad')
    axes[0,1].set_xlabel('Tipo de Propiedad')
    axes[0,1].set_ylabel('Precio por m² (DKK)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Tamaño promedio por tipo
    avg_size = df.groupby('house_type')['sqm'].mean().sort_values(ascending=True)
    avg_size.plot(kind='barh', ax=axes[1,0], color='lightcoral')
    axes[1,0].set_title('Tamaño Promedio por Tipo de Propiedad')
    axes[1,0].set_xlabel('Superficie (m²)')
    
    # 4. Volumen de transacciones por tipo
    transaction_volume = df['house_type'].value_counts()
    transaction_volume.plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
    axes[1,1].set_title('Volumen de Transacciones por Tipo')
    axes[1,1].set_ylabel('')
    
    plt.tight_layout()
    plt.show()

def visualize_market_behavior(df, market_analysis):
    """
    Visualizaciones para análisis de comportamiento de mercado.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset limpio
    market_analysis : dict
        Resultados del análisis de mercado
    """
    configure_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis del Comportamiento de Mercado', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Análisis por tipo de venta (si existe)
    if 'sales_type' in df.columns:
        df.boxplot(column=TARGET, by='sales_type', ax=axes[0,0])
        axes[0,0].set_title('Precios por Tipo de Venta')
        axes[0,0].set_xlabel('Tipo de Venta')
        axes[0,0].set_ylabel('Precio (DKK)')
    else:
        axes[0,0].text(0.5, 0.5, 'Datos de sales_type\nno disponibles', 
                      ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,0].set_title('Tipo de Venta - No Disponible')
    
    # 2. Distribución de cambios oferta vs compra (si existe)
    if '%_change_between_offer_and_purchase' in df.columns:
        df['%_change_between_offer_and_purchase'].hist(bins=50, ax=axes[0,1], alpha=0.7, color='skyblue')
        axes[0,1].axvline(0, color='red', linestyle='--', label='Sin cambio')
        axes[0,1].set_title('Distribución de Cambios Oferta vs Compra')
        axes[0,1].set_xlabel('% Cambio')
        axes[0,1].set_ylabel('Frecuencia')
        axes[0,1].legend()
    else:
        axes[0,1].text(0.5, 0.5, 'Datos de cambio\noferta-compra\nno disponibles', 
                      ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Cambio Oferta-Compra - No Disponible')
    
    # 3. Análisis por trimestre (si existe)
    if 'quarter' in df.columns:
        quarterly_prices = df.groupby('quarter')[TARGET].median()
        quarterly_prices.plot(kind='bar', ax=axes[1,0], color='lightgreen')
        axes[1,0].set_title('Precio Mediano por Trimestre')
        axes[1,0].set_xlabel('Trimestre')
        axes[1,0].set_ylabel('Precio Mediano (DKK)')
        axes[1,0].tick_params(axis='x', rotation=0)
    else:
        axes[1,0].text(0.5, 0.5, 'Datos de trimestre\nno disponibles', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Análisis Trimestral - No Disponible')
    
    # 4. Volumen de transacciones por trimestre (si existe)
    if 'quarter' in df.columns:
        quarterly_volume = df['quarter'].value_counts().sort_index()
        quarterly_volume.plot(kind='bar', ax=axes[1,1], color='orange')
        axes[1,1].set_title('Volumen de Transacciones por Trimestre')
        axes[1,1].set_xlabel('Trimestre')
        axes[1,1].set_ylabel('Número de Transacciones')
        axes[1,1].tick_params(axis='x', rotation=0)
    else:
        axes[1,1].text(0.5, 0.5, 'Datos de trimestre\nno disponibles', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Volumen Trimestral - No Disponible')
    
    plt.tight_layout()
    plt.show()

def visualize_market_segmentation(df):
    """
    Visualizaciones para análisis de segmentación de mercado.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset limpio
    """
    configure_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Segmentación de Mercado', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Segmentación por precio
    q25 = df[TARGET].quantile(0.25)
    q75 = df[TARGET].quantile(0.75)
    q90 = df[TARGET].quantile(0.90)
    
    df_temp = df.copy()
    df_temp['price_segment'] = pd.cut(
        df_temp[TARGET],
        bins=[0, q25, q75, q90, float('inf')],
        labels=['Económico', 'Medio', 'Alto', 'Premium']
    )
    
    segment_counts = df_temp['price_segment'].value_counts()
    segment_counts.plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%')
    axes[0,0].set_title('Distribución por Segmento de Precio')
    axes[0,0].set_ylabel('')
    
    # 2. Precio por segmento
    df_temp.boxplot(column=TARGET, by='price_segment', ax=axes[0,1])
    axes[0,1].set_title('Distribución de Precios por Segmento')
    axes[0,1].set_xlabel('Segmento')
    axes[0,1].set_ylabel('Precio (DKK)')
    
    # 3. Segmentación por antigüedad (si year_build está disponible)
    if 'year_build' in df.columns:
        current_year = 2024
        df_temp['property_age'] = current_year - df_temp['year_build']
        df_temp['age_category'] = pd.cut(
            df_temp['property_age'],
            bins=[0, 10, 25, 50, 100, float('inf')],
            labels=['Nueva', 'Moderna', 'Madura', 'Antigua', 'Histórica']
        )
        
        age_counts = df_temp['age_category'].value_counts()
        age_counts.plot(kind='bar', ax=axes[1,0], color='lightcoral')
        axes[1,0].set_title('Distribución por Antigüedad')
        axes[1,0].set_xlabel('Categoría de Antigüedad')
        axes[1,0].set_ylabel('Número de Propiedades')
        axes[1,0].tick_params(axis='x', rotation=45)
    else:
        axes[1,0].text(0.5, 0.5, 'Datos de año de\nconstrucción\nno disponibles', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Antigüedad - No Disponible')
    
    # 4. Precio por m² vs antigüedad (si year_build está disponible)
    if 'year_build' in df.columns:
        df_temp.boxplot(column='sqm_price', by='age_category', ax=axes[1,1])
        axes[1,1].set_title('Precio por m² según Antigüedad')
        axes[1,1].set_xlabel('Categoría de Antigüedad')
        axes[1,1].set_ylabel('Precio por m² (DKK)')
        axes[1,1].tick_params(axis='x', rotation=45)
    else:
        axes[1,1].text(0.5, 0.5, 'Datos de año de\nconstrucción\nno disponibles', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Precio/m² por Antigüedad - No Disponible')
    
    plt.tight_layout()
    plt.show()

# 9. Función principal actualizada con nuevos análisis
def run_complete_descriptive_analysis_extended(df, include_visualizations=True):
    """
    Ejecuta análisis descriptivo completo extendido.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset limpio con precios de vivienda
    include_visualizations : bool
        Si incluir visualizaciones (default: True)
        
    Returns:
    --------
    dict : Diccionario con todos los resultados
    """
    all_results = {}
    
    # Análisis anteriores
    all_results.update(run_complete_descriptive_analysis(df, include_visualizations=False))
    
    # Nuevos análisis
    print("\n" + "="*80)
    all_results['property_types'] = analyze_property_types(df)
    
    print("\n" + "="*80)
    all_results['market_behavior'] = analyze_market_behavior(df)
    
    print("\n" + "="*80)
    all_results['market_segmentation'] = analyze_market_segmentation(df)
    
    # Visualizaciones si se solicitan
    if include_visualizations:
        print("\n" + "="*80)
        print("GENERANDO VISUALIZACIONES...")
        
        visualize_property_type_analysis(df, all_results['property_types'])
        visualize_market_behavior(df, all_results['market_behavior'])
        visualize_market_segmentation(df)
    
    return all_results
