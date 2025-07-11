import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, normaltest, shapiro
import warnings
import numpy as np


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



###############################################################  ANALISIS UNIVARIADO ################################################################################

# --- ANÁLISIS NUMÉRICO CONTINUO ---

def analyze_numeric_series(series):
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
    for col in cols:
        print(f"\nAnálisis de `{col}`:")
        s = analyze_numeric_series(df[col])
        for k, v in s.items():
            print(f"  - {k}: {v:,.2f}" if isinstance(v, float) else f"  - {k}: {v}")
        plot_numeric_distribution(df[col], col)
        print("="*50)

# --- ANÁLISIS DISCRETO O CATEGÓRICO ---

def plot_discrete_distribution(series, col):
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
    """
    generar_wordcloud(df, col, max_words=50)
    
    return

def analizar_variable_categorica(df, col, top_n=10, other_threshold=0.8):
    """
    Visualiza y analiza una variable categórica con top_n categorías como máximo (otras se agrupan).
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
    """
    for col in categorical_cols:
        if col in df.columns:
            analizar_variable_categorica(df, col, top_n=5, other_threshold=0.8)
        else:
            print(f"Columna '{col}' no encontrada en el DataFrame.")

# --- FUNCIÓN PRINCIPAL ---

def run_univariate_analysis(df, continuous_cols=[], discrete_cols=[], categorical_cols=[]):
    if continuous_cols:
        describe_numeric(df, continuous_cols)
    if discrete_cols:
        describe_discrete(df, discrete_cols)
    if categorical_cols:
        plot_categorical_distributions(df, categorical_cols)

def enhanced_univariate_analysis(df, column, column_type='numeric'):
    """
    Análisis univariado mejorado con insights de negocio
    """

    
    warnings.filterwarnings('ignore')
    
    print(f"\n{'='*80}")
    print(f"ANÁLISIS UNIVARIADO MEJORADO: {column.upper()}")
    print(f"{'='*80}")
    
    if column_type == 'numeric':
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
        
        # Coeficiente de variación
        cv = (desc_stats['std'] / desc_stats['mean']) * 100
        print(f"   • Coeficiente de variación: {cv:.2f}%")
        if cv < 15:
            cv_interp = "Baja variabilidad"
        elif cv < 30:
            cv_interp = "Variabilidad moderada"
        else:
            cv_interp = "Alta variabilidad"
        print(f"     → Interpretación: {cv_interp}")
        
        # Detección de outliers
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
        
        # Tests de normalidad
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
        
        # Crear visualizaciones mejoradas
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
        
        # Insights de negocio
        print(f"\nINSIGHTS DE NEGOCIO:")
        
        # Concentración de datos
        perc_90 = np.percentile(df[column].dropna(), 90)
        perc_10 = np.percentile(df[column].dropna(), 10)
        concentration = (perc_90 - perc_10) / (desc_stats['max'] - desc_stats['min'])
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
        
        if jb_p < 0.05:
            print(f"   • Distribución no normal - considerar:")
            print(f"     → Transformaciones para normalizar")
            print(f"     → Uso de métodos no paramétricos")
        
        return {
            'descriptive_stats': desc_stats,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'cv': cv,
            'outliers_count': len(outliers),
            'normality_jb_p': jb_p,
            'insights': {
                'concentration': concentration,
                'high_variability': cv > 50,
                'needs_transformation': abs(skewness) > 1,
                'outlier_treatment_needed': len(outliers) > len(df) * 0.05
            }
        }
    
    else:  # Análisis para variables categóricas
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
        
        # Insights de negocio para categóricas
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

def create_univariate_summary(df, analyses_results=None):
    """
    Crea un resumen consolidado del análisis univariado con insights de negocio
    """    
    print("="*100)
    print("RESUMEN CONSOLIDADO DEL ANÁLISIS UNIVARIADO")
    print("="*100)
    
    # Análisis general del dataset
    print(f"\nPANORAMA GENERAL DEL DATASET:")
    print(f"   • Total de observaciones: {len(df):,}")
    print(f"   • Total de variables: {len(df.columns)}")
    print(f"   • Variables numéricas: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   • Variables categóricas: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Calidad de datos
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    print(f"\nCALIDAD DE DATOS:")
    print(f"   • Variables con valores faltantes: {len(missing_cols)}")
    print(f"   • Porcentaje total de datos faltantes: {(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%")
    
    # Análisis de variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nANÁLISIS DE VARIABLES NUMÉRICAS:")
    
    # Estadísticas de asimetría
    high_skew_vars = []
    transformation_needed = []
    
    for col in numeric_cols:
        skew = df[col].skew()
        if abs(skew) > 1:
            high_skew_vars.append((col, skew))
        if abs(skew) > 2:
            transformation_needed.append((col, skew))
    
    print(f"   • Variables con alta asimetría (|skew| > 1): {len(high_skew_vars)}")
    for var, skew in high_skew_vars[:5]:  # Top 5
        print(f"     → {var}: {skew:.2f}")
    
    print(f"   • Variables que requieren transformación (|skew| > 2): {len(transformation_needed)}")
    for var, skew in transformation_needed:
        print(f"     → {var}: {skew:.2f}")
    
    # Análisis de outliers
    outlier_summary = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = len(outliers)
    
    high_outlier_vars = [(k, v) for k, v in outlier_summary.items() if v > len(df) * 0.05]
    print(f"\nDETECCIÓN DE OUTLIERS:")
    print(f"   • Variables con >5% de outliers: {len(high_outlier_vars)}")
    for var, count in high_outlier_vars:
        print(f"     → {var}: {count:,} outliers ({count/len(df)*100:.1f}%)")
    
    # Análisis de variables categóricas
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nANÁLISIS DE VARIABLES CATEGÓRICAS:")
    
    high_cardinality = []
    low_diversity = []
    
    for col in cat_cols:
        unique_count = df[col].nunique()
        if unique_count > 50:
            high_cardinality.append((col, unique_count))
        
        # Cálculo de diversidad Shannon
        proportions = df[col].value_counts(normalize=True)
        shannon_div = -sum(proportions * np.log(proportions))
        if shannon_div < 1:
            low_diversity.append((col, shannon_div))
    
    print(f"   • Variables con alta cardinalidad (>50 categorías): {len(high_cardinality)}")
    for var, count in high_cardinality:
        print(f"     → {var}: {count:,} categorías únicas")
    
    print(f"   • Variables con baja diversidad (Shannon < 1): {len(low_diversity)}")
    for var, div in low_diversity:
        print(f"     → {var}: {div:.3f}")
    
    # Insights de negocio específicos
    print(f"\nINSIGHTS DE NEGOCIO CLAVE:")
    
    # Análisis de precios
    if 'purchase_price' in df.columns:
        price_median = df['purchase_price'].median()
        price_mean = df['purchase_price'].mean()
        price_std = df['purchase_price'].std()
        cv_price = (price_std / price_mean) * 100
        
        print(f"   • PRECIOS DE VIVIENDAS:")
        print(f"     → Mediana: ${price_median:,.0f}")
        print(f"     → Media: ${price_mean:,.0f}")
        print(f"     → Coeficiente de variación: {cv_price:.1f}%")
        
        if cv_price > 50:
            print(f"     → Alta variabilidad en precios - mercado diversificado")
        
        # Análisis de precios por m2
        if 'sqm_price' in df.columns:
            sqm_price_median = df['sqm_price'].median()
            print(f"     → Precio mediano por m²: ${sqm_price_median:,.0f}")
    
    # Análisis de características físicas
    if 'sqm' in df.columns:
        sqm_median = df['sqm'].median()
        sqm_q75 = df['sqm'].quantile(0.75)
        print(f"   • CARACTERÍSTICAS FÍSICAS:")
        print(f"     → Tamaño mediano: {sqm_median:.0f} m²")
        print(f"     → 75% de propiedades ≤ {sqm_q75:.0f} m²")
    
    if 'no_rooms' in df.columns:
        rooms_mode = df['no_rooms'].mode().iloc[0]
        rooms_median = df['no_rooms'].median()
        print(f"     → Número de habitaciones más común: {rooms_mode}")
        print(f"     → Mediana de habitaciones: {rooms_median}")
    
    # Análisis geográfico
    if 'region' in df.columns:
        region_counts = df['region'].value_counts()
        top_region = region_counts.index[0]
        top_region_pct = region_counts.iloc[0] / len(df) * 100
        print(f"   • DISTRIBUCIÓN GEOGRÁFICA:")
        print(f"     → Región dominante: {top_region} ({top_region_pct:.1f}%)")
        print(f"     → Número de regiones: {df['region'].nunique()}")
    
    # Análisis temporal
    if 'year_build' in df.columns:
        year_median = df['year_build'].median()
        year_q25 = df['year_build'].quantile(0.25)
        year_q75 = df['year_build'].quantile(0.75)
        print(f"   • CARACTERÍSTICAS TEMPORALES:")
        print(f"     → Año de construcción mediano: {year_median:.0f}")
        print(f"     → 50% de propiedades construidas entre {year_q25:.0f} y {year_q75:.0f}")
    
    # Recomendaciones estratégicas
    print(f"\nRECOMENDACIONES ESTRATÉGICAS:")
    
    print(f"   • PREPROCESSING:")
    if transformation_needed:
        print(f"     → Aplicar transformaciones logarítmicas a variables con alta asimetría")
    if high_outlier_vars:
        print(f"     → Implementar técnicas de tratamiento de outliers (winsorizing, capping)")
    if high_cardinality:
        print(f"     → Considerar agrupación de categorías para variables de alta cardinalidad")
    
    print(f"   • MODELADO:")
    if any(abs(df[col].skew()) > 2 for col in numeric_cols):
        print(f"     → Considerar modelos robustos a outliers (Random Forest, Gradient Boosting)")
    print(f"     → Aplicar técnicas de feature engineering para variables categóricas")
    print(f"     → Considerar interacciones entre variables geográficas y de precio")
    
    print(f"   • NEGOCIO:")
    if 'purchase_price' in df.columns and df['purchase_price'].std() / df['purchase_price'].mean() > 0.5:
        print(f"     → Segmentar mercado por rangos de precio para análisis específicos")
    print(f"     → Considerar análisis geográfico detallado para pricing estratégico")
    print(f"     → Implementar análisis temporal para identificar tendencias estacionales")
    
    return {
        'total_observations': len(df),
        'numeric_variables': len(numeric_cols),
        'categorical_variables': len(df.select_dtypes(include=['object']).columns),
        'high_skew_vars': len(high_skew_vars),
        'transformation_needed': len(transformation_needed),
        'high_outlier_vars': len(high_outlier_vars),
        'high_cardinality': len(high_cardinality),
        'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }

def create_advanced_univariate_dashboard(df):
    """
    Crea un dashboard avanzado con múltiples visualizaciones
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Distribución de precios con múltiples estadísticas
    ax1 = fig.add_subplot(gs[0, :2])
    df['purchase_price'].hist(bins=50, alpha=0.7, color='skyblue', ax=ax1)
    ax1.axvline(df['purchase_price'].mean(), color='red', linestyle='--', label=f'Media: ${df["purchase_price"].mean():,.0f}')
    ax1.axvline(df['purchase_price'].median(), color='green', linestyle='--', label=f'Mediana: ${df["purchase_price"].median():,.0f}')
    ax1.set_title('Distribución de Precios de Viviendas', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Precio de Compra ($)')
    ax1.set_ylabel('Frecuencia')
    ax1.legend()
    
    # 2. Análisis de outliers - Box plot conjunto
    ax2 = fig.add_subplot(gs[0, 2:])
    key_numeric = ['purchase_price', 'sqm_price', 'sqm', 'no_rooms']
    available_numeric = [col for col in key_numeric if col in df.columns]
    
    if available_numeric:
        # Normalizar datos para visualización comparativa
        df_norm = df[available_numeric].copy()
        for col in available_numeric:
            df_norm[col] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()
        
        df_norm.boxplot(ax=ax2, rot=45)
        ax2.set_title('Detección de Outliers (Datos Normalizados)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Valores Normalizados')
    
    # 3. Análisis de asimetría
    ax3 = fig.add_subplot(gs[1, :2])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skewness_data = [(col, df[col].skew()) for col in numeric_cols if col in df.columns]
    skewness_data.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if skewness_data:
        cols, skews = zip(*skewness_data[:10])  # Top 10
        colors = ['red' if abs(s) > 2 else 'orange' if abs(s) > 1 else 'green' for s in skews]
        bars = ax3.barh(cols, skews, color=colors, alpha=0.7)
        ax3.set_title('Análisis de Asimetría por Variable', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Asimetría (Skewness)')
        ax3.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax3.axvline(1, color='orange', linestyle='--', alpha=0.5, label='Moderada')
        ax3.axvline(-1, color='orange', linestyle='--', alpha=0.5)
        ax3.axvline(2, color='red', linestyle='--', alpha=0.5, label='Alta')
        ax3.axvline(-2, color='red', linestyle='--', alpha=0.5)
        ax3.legend()
    
    # 4. Análisis de correlación con variable objetivo
    ax4 = fig.add_subplot(gs[1, 2:])
    if 'purchase_price' in df.columns:
        correlations = df.corr()['purchase_price'].drop('purchase_price').abs().sort_values(ascending=False)
        correlations.head(10).plot(kind='bar', ax=ax4, color='lightcoral', alpha=0.7)
        ax4.set_title('Correlación con Precio de Compra', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Variables')
        ax4.set_ylabel('Correlación Absoluta')
        ax4.tick_params(axis='x', rotation=45)
    
    # 5. Análisis de variables categóricas - Diversidad
    ax5 = fig.add_subplot(gs[2, :2])
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    diversity_data = []
    
    for col in cat_cols:
        if col in df.columns:
            proportions = df[col].value_counts(normalize=True)
            shannon_div = -sum(proportions * np.log(proportions + 1e-10))  # Evitar log(0)
            diversity_data.append((col, shannon_div))
    
    if diversity_data:
        diversity_data.sort(key=lambda x: x[1], reverse=True)
        cols, divs = zip(*diversity_data)
        ax5.bar(cols, divs, color='lightgreen', alpha=0.7)
        ax5.set_title('Diversidad de Variables Categóricas (Shannon)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Variables')
        ax5.set_ylabel('Índice de Diversidad')
        ax5.tick_params(axis='x', rotation=45)
    
    # 6. Análisis temporal si existe
    ax6 = fig.add_subplot(gs[2, 2:])
    if 'year_build' in df.columns:
        year_counts = df['year_build'].value_counts().sort_index()
        year_counts.plot(kind='line', ax=ax6, color='navy', linewidth=2)
        ax6.set_title('Distribución Temporal de Construcciones', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Año de Construcción')
        ax6.set_ylabel('Número de Propiedades')
        ax6.grid(True, alpha=0.3)
    
    # 7. Análisis de calidad de datos
    ax7 = fig.add_subplot(gs[3, :2])
    missing_data = df.isnull().sum()[df.isnull().sum() > 0]
    if len(missing_data) > 0:
        missing_data.plot(kind='bar', ax=ax7, color='orange', alpha=0.7)
        ax7.set_title('Valores Faltantes por Variable', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Variables')
        ax7.set_ylabel('Cantidad de Valores Faltantes')
        ax7.tick_params(axis='x', rotation=45)
    else:
        ax7.text(0.5, 0.5, 'No hay valores faltantes', ha='center', va='center', 
                transform=ax7.transAxes, fontsize=12, fontweight='bold')
        ax7.set_title('Calidad de Datos', fontsize=14, fontweight='bold')
    
    # 8. Métricas de resumen
    ax8 = fig.add_subplot(gs[3, 2:])
    ax8.axis('off')
    
    # Calcular métricas clave
    total_obs = len(df)
    numeric_vars = len(df.select_dtypes(include=[np.number]).columns)
    cat_vars = len(df.select_dtypes(include=['object']).columns)
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    
    summary_text = f"""
    RESUMEN EJECUTIVO
    
    Datos Generales:
    • Total observaciones: {total_obs:,}
    • Variables numéricas: {numeric_vars}
    • Variables categóricas: {cat_vars}
    • Datos faltantes: {missing_pct:.1f}%
    
    Precios (si disponible):
    • Media: ${df['purchase_price'].mean():,.0f if 'purchase_price' in df.columns else 'N/A'}
    • Mediana: ${df['purchase_price'].median():,.0f if 'purchase_price' in df.columns else 'N/A'}
    • Desv. Estándar: ${df['purchase_price'].std():,.0f if 'purchase_price' in df.columns else 'N/A'}
    
    Calidad de Datos:
    • Variables normales: {sum(1 for col in numeric_cols if abs(df[col].skew()) < 0.5)}
    • Variables asimétricas: {sum(1 for col in numeric_cols if abs(df[col].skew()) > 1)}
    • Variables con outliers: {sum(1 for col in numeric_cols if len(df[(df[col] < df[col].quantile(0.25) - 1.5*(df[col].quantile(0.75)-df[col].quantile(0.25))) | (df[col] > df[col].quantile(0.75) + 1.5*(df[col].quantile(0.75)-df[col].quantile(0.25)))]) > len(df)*0.05)}
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=11, 
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('Dashboard Avanzado de Análisis Univariado', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()