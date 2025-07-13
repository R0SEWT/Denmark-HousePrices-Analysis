"""
Módulo para el enriquecimiento con datos geoespaciales.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def add_geospatial_features(df: pd.DataFrame, postal_data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Agrega características geoespaciales al dataset.
    
    Args:
        df: DataFrame con datos de propiedades
        postal_data_path: Ruta opcional a datos de códigos postales
    
    Returns:
        DataFrame enriquecido con características geográficas
    """
    df_result = df.copy()
    
    # Placeholder para características geográficas sintéticas
    # En un escenario real, esto cargaría datos reales de códigos postales,
    # coordenadas, distancias a centros urbanos, etc.
    
    # Generar variables geográficas simuladas basadas en región
    if 'region' in df_result.columns:
        # Mapear regiones a características urbanas simuladas
        urban_density_map = {
            'Copenhagen': 5, 'Aarhus': 4, 'Odense': 3, 'Aalborg': 3,
            'Frederiksberg': 5, 'Esbjerg': 2, 'Randers': 2, 'Kolding': 2
        }
        
        # Densidad urbana (1-5, donde 5 es más urbano)
        df_result['urban_density'] = df_result['region'].map(urban_density_map).fillna(1)
        
        # Distancia simulada al centro (basada en región)
        center_distance_map = {
            'Copenhagen': 10, 'Aarhus': 15, 'Odense': 20, 'Aalborg': 25,
            'Frederiksberg': 5, 'Esbjerg': 35, 'Randers': 30, 'Kolding': 25
        }
        df_result['distance_to_center'] = df_result['region'].map(center_distance_map).fillna(50)
        
        # Variables categóricas geográficas
        df_result['location_type'] = df_result['urban_density'].apply(
            lambda x: 'Urban' if x >= 4 else 'Suburban' if x >= 2 else 'Rural'
        )
        
        # Acceso a transporte (simulado)
        df_result['transport_access'] = df_result['urban_density'] * 0.8 + np.random.normal(0, 0.2, len(df_result))
        df_result['transport_access'] = np.clip(df_result['transport_access'], 1, 5)
    
    print(f"Características geográficas agregadas: {['urban_density', 'distance_to_center', 'location_type', 'transport_access']}")
    return df_result

def create_geographic_clusters(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    Crea clusters geográficos basados en características de ubicación.
    
    Args:
        df: DataFrame con características geográficas
        n_clusters: Número de clusters a crear
    
    Returns:
        DataFrame con asignaciones de cluster
    """
    df_result = df.copy()
    
    # Clustering simple basado en características geográficas existentes
    if all(col in df_result.columns for col in ['urban_density', 'distance_to_center']):
        try:
            from sklearn.cluster import KMeans
            
            features = df_result[['urban_density', 'distance_to_center']].fillna(0)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df_result['geo_cluster'] = kmeans.fit_predict(features)
            
            print(f"Clusters geográficos creados: {n_clusters} clusters")
        except ImportError:
            # Si sklearn no está disponible, crear clusters simples basados en rangos
            print("sklearn no disponible, creando clusters simples...")
            df_result['geo_cluster'] = pd.qcut(
                df_result['urban_density'] + df_result['distance_to_center'], 
                q=n_clusters, 
                labels=False, 
                duplicates='drop'
            )
    
    return df_result
