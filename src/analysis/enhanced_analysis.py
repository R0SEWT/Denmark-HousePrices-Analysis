"""
Módulo de análisis univariado avanzado
Contiene funciones para análisis detallado con insights de negocio
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, normaltest, shapiro
import numpy as np
import warnings


def enhanced_univariate_analysis(df, column, column_type='numeric'):
    """
    Análisis univariado mejorado con insights de negocio
    
    Args:
        df: DataFrame
        column: Nombre de la columna a analizar
        column_type: Tipo de variable ('numeric' o 'categorical')
        
    Returns:
        Diccionario con resultados del análisis
    """
    warnings.filterwarnings('ignore')
    
    print(f"\n{'='*80}")
    print(f"ANÁLISIS UNIVARIADO MEJORADO: {column.upper()}")
    print(f"{'='*80}")
    
    if column_type == 'numeric':
        return _analyze_numeric_enhanced(df, column)
    else:
        return _analyze_categorical_enhanced(df, column)


def _analyze_numeric_enhanced(df, column):
    """Análisis avanzado para variables numéricas"""
    
    # Estadísticas descriptivas básicas
    desc_stats = df[column].describe()
    print(f"\nESTADÍSTICAS DESCRIPTIVAS:")
    print(f"   • Observaciones: {len(df[column]):,}")
    print(f"   • Valores únicos: {df[column].nunique():,}")
    print(f"   • Valores nulos: {df[column].isnull().sum():,} ({df[column].isnull().sum()/len(df)*100:.2f}%)")
    print(f"   • Mínimo: {desc_stats['min']:,.2f}")
    print(f"   • Q1 (25%): {desc_stats['25%']:,.2f}")
    print(f"   • Mediana: {desc_stats['50%']:,.2f}")
    print(f"   • Q3 (75%): {desc_stats['75%']:,.2f}")
    print(f"   • Máximo: {desc_stats['max']:,.2f}")
    print(f"   • Media: {desc_stats['mean']:,.2f}")
    print(f"   • Desviación estándar: {desc_stats['std']:,.2f}")
    
    # Medidas de forma
    skewness = df[column].skew()
    kurtosis = df[column].kurtosis()
    _print_shape_measures(skewness, kurtosis)
    
    # Coeficiente de variación
    cv = (desc_stats['std'] / desc_stats['mean']) * 100
    _print_coefficient_variation(cv)
    
    # Detección de outliers
    outliers = _detect_outliers(df, column, desc_stats)
    
    # Tests de normalidad
    _perform_normality_tests(df, column)
    
    # Crear visualizaciones
    _create_numeric_visualizations(df, column, desc_stats)
    
    # Insights de negocio
    _print_numeric_insights(df, column, desc_stats, skewness, cv, outliers)
    
    return {
        'descriptive_stats': desc_stats,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'cv': cv,
        'outliers_count': len(outliers),
        'insights': {
            'concentration': _calculate_concentration(df, column),
            'high_variability': cv > 50,
            'needs_transformation': abs(skewness) > 1,
            'outlier_treatment_needed': len(outliers) > len(df) * 0.05
        }
    }


def _analyze_categorical_enhanced(df, column):
    """Análisis avanzado para variables categóricas"""
    
    value_counts = df[column].value_counts()
    prop_counts = df[column].value_counts(normalize=True)
    
    print(f"\nESTADÍSTICAS DESCRIPTIVAS:")
    print(f"   • Observaciones: {len(df[column]):,}")
    print(f"   • Valores únicos: {df[column].nunique():,}")
    print(f"   • Valores nulos: {df[column].isnull().sum():,} ({df[column].isnull().sum()/len(df)*100:.2f}%)")
    print(f"   • Moda: {df[column].mode().iloc[0] if not df[column].mode().empty else 'N/A'}")
    
    # Concentración de categorías
    top_category_prop = prop_counts.iloc[0]
    print(f"   • Concentración en categoría principal: {top_category_prop:.2%}")
    
    # Índice de diversidad (Shannon)
    shannon_diversity = -sum(prop_counts * np.log(prop_counts))
    print(f"   • Índice de diversidad (Shannon): {shannon_diversity:.3f}")
    
    print(f"\nDISTRIBUCIÓN DE CATEGORÍAS:")
    for i, (cat, count) in enumerate(value_counts.head(10).items()):
        print(f"   • {cat}: {count:,} ({prop_counts.iloc[i]:.2%})")
    
    # Visualización mejorada
    _create_categorical_visualizations(df, column, value_counts, prop_counts)
    
    # Insights de negocio
    _print_categorical_insights(df, column, top_category_prop, shannon_diversity)
    
    return {
        'value_counts': value_counts,
        'prop_counts': prop_counts,
        'shannon_diversity': shannon_diversity,
        'top_category_prop': top_category_prop,
        'insights': {
            'high_concentration': top_category_prop > 0.5,
            'high_cardinality': df[column].nunique() > 50,
            'needs_grouping': df[column].nunique() > 20
        }
    }


def _print_shape_measures(skewness, kurtosis):
    """Imprime e interpreta medidas de forma"""
    print(f"\nMEDIDAS DE FORMA:")
    print(f"   • Asimetría (Skewness): {skewness:.3f}")
    if abs(skewness) < 0.5:
        skew_interp = "Aproximadamente simétrica"
    elif skewness > 0.5:
        skew_interp = "Asimetría positiva (cola derecha)"
    else:
        skew_interp = "Asimetría negativa (cola izquierda)"
    print(f"     → Interpretación: {skew_interp}")
    
    print(f"   • Curtosis: {kurtosis:.3f}")
    if kurtosis > 0:
        kurt_interp = "Leptocúrtica (más puntiaguda que normal)"
    elif kurtosis < 0:
        kurt_interp = "Platicúrtica (más aplanada que normal)"
    else:
        kurt_interp = "Mesocúrtica (similar a normal)"
    print(f"     → Interpretación: {kurt_interp}")


def _print_coefficient_variation(cv):
    """Imprime e interpreta el coeficiente de variación"""
    print(f"   • Coeficiente de variación: {cv:.2f}%")
    if cv < 15:
        cv_interp = "Baja variabilidad"
    elif cv < 30:
        cv_interp = "Variabilidad moderada"
    else:
        cv_interp = "Alta variabilidad"
    print(f"     → Interpretación: {cv_interp}")


def _detect_outliers(df, column, desc_stats):
    """Detecta outliers usando el método IQR"""
    Q1 = desc_stats['25%']
    Q3 = desc_stats['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    print(f"\nDETECCIÓN DE OUTLIERS (Método IQR):")
    print(f"   • Límite inferior: {lower_bound:.2f}")
    print(f"   • Límite superior: {upper_bound:.2f}")
    print(f"   • Total de outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")
    
    if len(outliers) > 0:
        print(f"   • Outliers más extremos:")
        extreme_outliers = outliers.nlargest(3).tolist() + outliers.nsmallest(3).tolist()
        for val in set(extreme_outliers):
            print(f"     → {val:,.2f}")
    
    return outliers


def _perform_normality_tests(df, column):
    """Realiza tests de normalidad"""
    print(f"\nTESTS DE NORMALIDAD:")
    
    # Jarque-Bera test
    jb_stat, jb_p = jarque_bera(df[column].dropna())
    print(f"   • Jarque-Bera Test:")
    print(f"     → Estadístico: {jb_stat:.3f}")
    print(f"     → p-valor: {jb_p:.6f}")
    print(f"     → Resultado: {'Normal' if jb_p > 0.05 else 'No normal'}")
    
    # Shapiro-Wilk test (si la muestra es pequeña)
    if len(df[column].dropna()) <= 5000:
        sw_stat, sw_p = shapiro(df[column].dropna().sample(min(3000, len(df[column].dropna()))))
        print(f"   • Shapiro-Wilk Test (muestra de {min(3000, len(df[column].dropna()))} obs):")
        print(f"     → Estadístico: {sw_stat:.3f}")
        print(f"     → p-valor: {sw_p:.6f}")
        print(f"     → Resultado: {'Normal' if sw_p > 0.05 else 'No normal'}")


def _create_numeric_visualizations(df, column, desc_stats):
    """Crea visualizaciones para variables numéricas"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Análisis Univariado Completo: {column}', fontsize=16, fontweight='bold')
    
    # 1. Histograma con KDE
    axes[0,0].hist(df[column].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(desc_stats['mean'], color='red', linestyle='--', label=f'Media: {desc_stats["mean"]:.2f}')
    axes[0,0].axvline(desc_stats['50%'], color='green', linestyle='--', label=f'Mediana: {desc_stats["50%"]:.2f}')
    axes[0,0].set_title('Histograma con Estadísticas Clave')
    axes[0,0].set_xlabel(column)
    axes[0,0].set_ylabel('Frecuencia')
    axes[0,0].legend()
    
    # 2. Box plot mejorado
    bp = axes[0,1].boxplot(df[column].dropna(), patch_artist=True, 
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
    axes[0,1].set_title('Box Plot con Detección de Outliers')
    axes[0,1].set_ylabel(column)
    
    # 3. Violin plot
    axes[0,2].violinplot(df[column].dropna(), showmeans=True, showmedians=True)
    axes[0,2].set_title('Violin Plot - Distribución de Densidad')
    axes[0,2].set_ylabel(column)
    
    # 4. Q-Q plot
    stats.probplot(df[column].dropna(), dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot vs Normal')
    
    # 5. Density plot con transformaciones
    axes[1,1].hist(df[column].dropna(), bins=50, alpha=0.5, density=True, label='Original')
    if df[column].min() > 0:  # Solo si todos los valores son positivos
        log_data = np.log(df[column].dropna())
        axes[1,1].hist(log_data, bins=50, alpha=0.5, density=True, label='Log Transform')
    axes[1,1].set_title('Comparación: Original vs Transformada')
    axes[1,1].legend()
    
    # 6. Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    perc_values = [np.percentile(df[column].dropna(), p) for p in percentiles]
    axes[1,2].plot(percentiles, perc_values, 'o-', color='orange', linewidth=2, markersize=8)
    axes[1,2].set_title('Análisis de Percentiles')
    axes[1,2].set_xlabel('Percentil')
    axes[1,2].set_ylabel('Valor')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def _create_categorical_visualizations(df, column, value_counts, prop_counts):
    """Crea visualizaciones para variables categóricas"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Análisis Univariado: {column}', fontsize=16, fontweight='bold')
    
    # 1. Bar plot horizontal
    top_categories = value_counts.head(15)
    top_categories.plot(kind='barh', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Top 15 Categorías - Frecuencia')
    axes[0,0].set_xlabel('Frecuencia')
    
    # 2. Pie chart para top categorías
    top_5 = value_counts.head(5)
    other_sum = value_counts.iloc[5:].sum()
    if other_sum > 0:
        plot_data = pd.concat([top_5, pd.Series([other_sum], index=['Otros'])])
    else:
        plot_data = top_5
    
    axes[0,1].pie(plot_data.values, labels=plot_data.index, autopct='%1.1f%%', startangle=90)
    axes[0,1].set_title('Distribución Porcentual (Top 5 + Otros)')
    
    # 3. Pareto chart
    cumulative_prop = prop_counts.cumsum()
    ax_pareto = axes[1,0]
    ax_pareto.bar(range(len(value_counts.head(15))), value_counts.head(15).values, alpha=0.7)
    ax_pareto.set_xlabel('Categorías')
    ax_pareto.set_ylabel('Frecuencia')
    ax_pareto.set_title('Gráfico de Pareto')
    ax_pareto.tick_params(axis='x', rotation=45)
    
    ax_pareto2 = ax_pareto.twinx()
    ax_pareto2.plot(range(len(cumulative_prop.head(15))), cumulative_prop.head(15).values, 
                   color='red', marker='o', linewidth=2)
    ax_pareto2.set_ylabel('Proporción Acumulada')
    ax_pareto2.set_ylim(0, 1)
    
    # 4. Información estadística
    axes[1,1].axis('off')
    
    # Build concentration text based on available categories
    concentration_text = ""
    if len(cumulative_prop) >= 3:
        concentration_text += f"        • Top 3 categorías: {cumulative_prop.iloc[2]:.2%} de los datos\n"
    if len(cumulative_prop) >= 5:
        concentration_text += f"        • Top 5 categorías: {cumulative_prop.iloc[4]:.2%} de los datos\n"
    
    top_category_prop = prop_counts.iloc[0]
    shannon_diversity = -sum(prop_counts * np.log(prop_counts))
    
    info_text = f"""
    RESUMEN ESTADÍSTICO
    
    • Total de categorías: {df[column].nunique():,}
    • Categoría más frecuente: {value_counts.index[0]}
    • Proporción de la moda: {top_category_prop:.2%}
    • Índice de diversidad: {shannon_diversity:.3f}
    
    CONCENTRACIÓN:
{concentration_text}
    INSIGHTS:
    • {'Alta concentración' if top_category_prop > 0.5 else 'Distribución balanceada'}
    • {'Baja diversidad' if shannon_diversity < 1 else 'Diversidad moderada/alta'}
    """
    axes[1,1].text(0.1, 0.9, info_text, transform=axes[1,1].transAxes, 
                  fontsize=10, verticalalignment='top', 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def _calculate_concentration(df, column):
    """Calcula la concentración de datos"""
    perc_90 = np.percentile(df[column].dropna(), 90)
    perc_10 = np.percentile(df[column].dropna(), 10)
    desc_stats = df[column].describe()
    return (perc_90 - perc_10) / (desc_stats['max'] - desc_stats['min'])


def _print_numeric_insights(df, column, desc_stats, skewness, cv, outliers):
    """Imprime insights para variables numéricas"""
    print(f"\nINSIGHTS DE NEGOCIO:")
    
    # Concentración de datos
    concentration = _calculate_concentration(df, column)
    print(f"   • Concentración de datos: {concentration:.2f}")
    print(f"     → El 80% de los datos se concentra en {concentration*100:.1f}% del rango total")
    
    # Potencial de segmentación
    if cv > 50:
        print(f"   • Alta variabilidad sugiere potencial para segmentación")
    
    # Recomendaciones de transformación
    print(f"\nRECOMENDACIONES DE PREPROCESSING:")
    
    if len(outliers) > len(df) * 0.05:  # Más del 5% son outliers
        print(f"   • Considerar tratamiento de outliers (winsorizing, capping)")
    
    if abs(skewness) > 1:
        print(f"   • Considerar transformación para reducir asimetría:")
        if skewness > 0:
            print(f"     → Transformación logarítmica, Box-Cox, o raíz cuadrada")
        else:
            print(f"     → Transformación exponencial o potencial")


def _print_categorical_insights(df, column, top_category_prop, shannon_diversity):
    """Imprime insights para variables categóricas"""
    print(f"\nINSIGHTS DE NEGOCIO:")
    
    if top_category_prop > 0.8:
        print(f"   • Extrema concentración en una categoría - verificar calidad de datos")
    elif top_category_prop > 0.5:
        print(f"   • Alta concentración - considerar binning o agrupación")
    else:
        print(f"   • Distribución relativamente balanceada")
    
    if df[column].nunique() > 50:
        print(f"   • Alta cardinalidad - considerar:")
        print(f"     → Agrupación de categorías poco frecuentes")
        print(f"     → Encoding específico para ML")
