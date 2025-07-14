#  Análisis exploratorio y modelado predictivo de precios residenciales en Dinamarca (1992–2024)

Kaggle dataset: [Danish Residential Housing Prices 1992-2024](https://www.kaggle.com/datasets/martinfrederiksen/danish-residential-housing-prices-1992-2024/data) 


## 1. Descripción del caso de uso

El precio de la vivienda es un tema socioeconómico de gran relevancia, pues la compra de una casa suele ser la inversión más importante en la vida de una familia[1]. En Dinamarca, al igual que en muchos países, el mercado inmobiliario ha experimentado notables alzas de precios en las últimas décadas. Por ejemplo, el precio promedio de una vivienda unifamiliar aumentó **153,9% en términos reales** entre 1992 y 2020[2]. Estas fluctuaciones incluyen periodos de **boom** seguidos de ajustes: antes de la crisis financiera de 2008 los precios crecieron aceleradamente y luego cayeron \~20% para 2009, recuperándose en la década siguiente[2]. Este comportamiento cíclico ha generado interrogantes sobre la existencia de burbujas inmobiliarias y la sostenibilidad de los precios respecto a fundamentos económicos[2].

Dada la importancia de la vivienda tanto para la economía nacional como para el bienestar social, **analizar y predecir los precios de las viviendas** resulta fundamental. Una predicción precisa ayuda a compradores y vendedores a tomar decisiones informadas, a la vez que permite a planificadores y entidades financieras evaluar riesgos. Modelos de *machine learning* ya han mostrado eficacia en la predicción de precios inmobiliarios[4], pudiendo descubrir patrones ocultos en los datos históricos. En este proyecto, proponemos aplicar técnicas de Big Data y aprendizaje supervisado para **encontrar patrones y predecir el precio de viviendas residenciales en Dinamarca** usando datos históricos de 1992 a 2024. Se busca no solo alta precisión predictiva sino también **interpretabilidad**, de modo que los resultados brinden conocimiento claro sobre **qué factores influyen** en el precio (por ejemplo, ubicación, tamaño, antigüedad, etc.). Esto está alineado con la tendencia hacia **IA explicable**, utilizando métodos como SHAP o LIME para interpretar modelos complejos[5]. En resumen, el caso de uso se enfoca en demostrar cómo el análisis de grandes volúmenes de datos inmobiliarios puede apoyar la toma de decisiones en el mercado de la vivienda, un ámbito de gran impacto económico y social.

---
## 2. Descripción del conjunto de datos

El conjunto de datos utilizado proviene de la plataforma Kaggle (aporte de Martin Frederiksen, 2024) e incluye ~1,5 millones de registros de ventas de viviendas residenciales en Dinamarca, cubriendo el período 1992 a 2024. Cada fila representa una transacción inmobiliaria residencial real durante esos 32 años, recopiladas originalmente de registros oficiales de ventas. El dataset completo (`.parquet`) contiene aproximadamente **1.5 millones de registros** de ventas de viviendas residenciales en Dinamarca durante el período **1992 a 2024**.


### 2.1 Procedencia y recopilación

* Los datos fueron recolectados mediante técnicas de **web scraping**, ejecutadas sobre fuentes públicas como:

  * El portal inmobiliario **Boliga**.
  * Sitios oficiales de estadísticas danesas, como **Statistikbanken** y **Danmarks Statistik**.

* La recolección se llevó a cabo usando **scripts en Python**, ejecutados en notebooks Jupyter del repositorio público del autor.

![Fuentes primarias del dataset de kaggle](../utils/doc_src/fuentes_primarias.png)
_- Fuentes primarias del dataset de Kaggle (repositorio de Martin Frederiksen)_

### 2.2 Proceso de limpieza y estructuración

* Se descargaron más de **80 archivos CSV** comprimidos, ubicados en la carpeta *Housing\_data\_raw*, utilizando el notebook `Webscrape_script.ipynb`.

* Posteriormente, el notebook `BoligsalgConcatCleaningGit.ipynb` concatenó, depuró y estructuró los datos mediante:

  * Estandarización de formatos (fechas, precios, áreas).
  * Eliminación de valores inválidos o simbólicos (como guiones ‘–’).
  * Filtrado o imputación de datos faltantes según reglas definidas.

### 2.3 Enriquecimiento de variables

* A los datos transaccionales se integraron variables **macroeconómicas y geográficas**, tales como:

  * **Tasas de inflación e interés.**
  * **Datos hipotecarios históricos.**
  * **Códigos postales y regiones administrativas.**

* Estos datos complementarios se extrajeron de fuentes públicas adicionales y se incorporaron desde la carpeta *Additional\_data* del repositorio original.

### 2.4 Estructura final del dataset

* El resultado final consiste en **dos archivos `.parquet`** (`DKHousingprices_1` y `DKHousingprices_2`) que contienen:

  * Datos consolidados, limpios y estructurados.
  * Variables clave como: fecha de venta, precio, tipo de propiedad, superficie, número de habitaciones y ubicación.
  * Integración de contexto económico y geográfico para potenciar análisis predictivos y exploratorios.

[Link de repositorio del proceso de mineria y limpieza de datos llevado a cabo por Martin Frederiksen](https://github.com/MartinSamFred/Danish-residential-housingPrices-1992-2024)


---

## 🏷️ Columnas disponibles (Cleaned files)

| Nº  | Nombre columna                                 | Descripción                                                                                         | Observaciones                            |
|-----|------------------------------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------|
| 0   | `date`                                         | Fecha de la transacción                                                                             | —                                        |
| 1   | `quarter`                                      | Trimestre según calendario estándar                                                                 | —                                        |
| 2   | `house_id`                                     | ID único de vivienda                                                                                | Puede eliminarse                         |
| 3   | `house_type`                                   | Tipo de vivienda: `'Villa'`, `'Farm'`, `'Summerhouse'`, `'Apartment'`, `'Townhouse'`               | —                                        |
| 4   | `sales_type`                                   | Tipo de venta: `'regular_sale'`, `'family_sale'`, `'other_sale'`, `'auction'`, `'-'`              | `'-'` puede eliminarse                   |
| 5   | `year_build`                                   | Año de construcción (rango 1000–2024)                                                               | Se puede restringir más                  |
| 6   | `purchase_price`                               | Precio de compra en coronas danesas (DKK)                                                           | —                                        |
| 7   | `%_change_between_offer_and_purchase`          | Variación % entre precio ofertado y precio de compra                                                | Puede ser negativa, cero o positiva      |
| 8   | `no_rooms`                                     | Número de habitaciones                                                                              | —                                        |
| 9   | `sqm`                                          | Metros cuadrados                                                                                   | —                                        |
| 10  | `sqm_price`                                    | Precio por metro cuadrado (precio_compra / metros cuadrados)                                        | —                                        |
| 11  | `address`                                      | Dirección                                                                                           | —                                        |
| 12  | `zip_code`                                     | Código postal                                                                                       | —                                        |
| 13  | `city`                                         | Ciudad                                                                                              | —                                        |
| 14  | `area`                                         | Área geográfica: `'East & mid jutland'`, `'North jutland'`, `'Other islands'`, `'Copenhagen'`, etc. | —                                        |
| 15  | `region`                                       | Región: `'Jutland'`, `'Zealand'`, `'Fyn & islands'`, `'Bornholm'`                                   | —                                        |
| 16  | `nom_interest_rate%`                           | Tasa de interés nominal danesa por trimestre (no convertida a formato trimestral)                  | —                                        |
| 17  | `dk_ann_infl_rate%`                            | Tasa de inflación anual danesa por trimestre (no convertida)                                       | —                                        |
| 18  | `yield_on_mortgage_credit_bonds%`              | Tasa de bonos hipotecarios a 30 años (sin spread)                                                   | —                                        |
<p align="center">
  <img src="utils/doc_src/distribucion_de_categorias_por_tipo.png" alt="Figura V" />
</p>

<p align="center"><em>Figura V. Distribución de categorías por tipo</em></p>


Se observa que la mayoría de las columnas contienen datos **numéricos**, lo cual es favorable para su análisis y posterior modelado.


---

## 3. Enfoque metodológico

### Objetivo general

Desarrollar un análisis exploratorio (EDA) y un modelo predictivo explicable de los precios de viviendas residenciales en Dinamarca entre 1992 y 2024, utilizando técnicas de Big Data para identificar patrones, factores relevantes y posibles anomalías en el mercado inmobiliario. (cita al informe)

---

### Objetivos específicos

1. **Explorar y limpiar** el dataset de precios de viviendas, identificando valores atípicos y patrones generales.
2. **Analizar** de forma univariada y bivariada las variables clave (precios, metros cuadrados, ubicación, etc.).
3. **Determinar** relaciones entre variables que influyen significativamente en el precio de una vivienda.
4. **Construir** modelos supervisados de predicción de precios, priorizando precisión e interpretabilidad.
5. **Detectar** posibles anomalías estructurales en el mercado, como burbujas o rupturas de tendencia, usando análisis de residuales en series temporales.

---

### Preguntas orientadoras

* ¿Qué factores tienen mayor impacto en el precio de una vivienda en Dinamarca?
* ¿Qué diferencias existen entre regiones y tipos de vivienda?
* ¿Se pueden detectar cambios anómalos o inusuales en el mercado a lo largo del tiempo?
* ¿Qué tan precisas y explicables pueden ser las predicciones de precios usando modelos de ML?

---

### Metodología general

* **Tipo de estudio**: Cuantitativo, correlacional, longitudinal (1992–2024).
* **Enfoque**: Basado en ciencia de datos y aprendizaje automático.
* **Técnicas**:

  * Limpieza y transformación de datos con H2O/Pandas
  * EDA con análisis univariado, bivariado y visualización
  * Modelado predictivo con H2O AutoML, XGBoost y GLM
  * Interpretabilidad con SHAP o coeficientes
  * Detección de anomalías sobre residuales de series temporales

<p align="center">
  <img src="utils/doc_src/data_pipeline_overview.png" alt="Figura V" />
</p>

<p align="center"><em>Figura V. Data Pipeline para el análisis y predicción de precios de vivienda</em></p>



## Analisis de datos

<p align="center">
  <img src="utils/doc_src/data_analysis_flow_complete.png" alt="Figura V" />
</p>

<p align="center"><em>Figura X. Flujo de trabajo general del análisis de datos y predicción de precios con tareas proyectadas (TBD)</em></p>



### 3.2.1 Análisis exploratorio de los datos (EDA)

#### 3.2.1.1 Carga del dataset
<p align="center">
  <img src="utils/doc_src/cluster_visualise.png" alt="Figura V" />
</p>

<p align="center"><em>Figura X. Iniciacion de cluster H2O</em></p>

* Se utilizó el dataset completo de precios de viviendas, que contiene aproximadamente **1.5 millones de registros** y **19 columnas** relevantes para el análisis.

Para lograrlo, se realizó una carga distribuida del dataset en un clúster H2O con dos nodos de cómputo, lo que permitió manejar eficientemente el volumen de datos y realizar análisis complejos sin comprometer el rendimiento.

<p align="center">
  <img src="utils/doc_src/cluster.png" alt="Figura V" />
</p>

<p align="center"><em>Figura X. Inicialización del clúster distribuido en H2O</em></p>

Los datos fueron cargados mediante `h2o.import_file()`, una función que permite leer grandes volúmenes en memoria distribuida. Para ello, se habilitó una carpeta compartida en el servidor utilizando **Samba**, la cual fue montada como directorio de trabajo accesible por todos los nodos del clúster H2O.


* El clúster se configuró con dos nodos conectados con las siguientes especificaciones:

| Nodo   | CPU               | RAM          | GPU                  |
|--------|------------------|--------------|-----------------------|
| Nodo 1 | Intel i5-12600K  | 16 GB DDR4   | RTX 4060 (8 GB)       |
| Nodo 2 | AMD Ryzen 5 7600X| 16 GB DDR5   | RTX 4060 Ti (16 GB)   |


<p align="center">
  <img src="utils/doc_src/carga_inicial.png" alt="Figura V" />
</p>

<p align="center"><em>Figura X. Inicialización del clúster distribuido en H2O</em></p>


*Resumen del dataset: número de registros, columnas y dimensiones generales.*

*Análisis del uso de memoria.*

* Se valida que el tamaño del dataset es considerable, pero no excede la capacidad de carga en memoria disponible.

* El conjunto presenta una estructura manejable desde el punto de vista computacional, a pesar de su volumen.




#### 3.2.1.2 Análisis preliminar de los datos

h2o.describe(chunk_summary=True) permite obtener un resumen estadístico de las variables numéricas, incluyendo conteos, medias, desviaciones estándar, valores mínimos y máximos, zeros y valores faltantes, asi como una pequeña muestra (`head`) de los datos.


| Column       | Type   | Min        | Max        | Mean        | Std Dev       | Missing | Zeros |
|--------------|--------|------------|------------|-------------|---------------|---------|--------|
| `date`       | int    | 6.95e+17   | 1.73e+18   | 1.35e+18     | 2.85e+17       | 0       | 0      |
| `quarter`    | int    | 88         | 219        | 170.70       | 36.18          | 0       | 0      |
| `house_id`   | int    | 0          | 15,079,070 | 753,953.5    | 435,295.7      | 0       | 1      |

...
---

##### 3.2.1.2.1 Observaciones iniciales

El análisis descriptivo permite identificar algunas variables con valores atípicos o inconsistencias que podrían afectar el modelo si no se tratan adecuadamente:

* **`%_change_between_offer_and_purchase`**
  Contiene valores negativos y **966,554 ceros (\~64%)**. Posibles explicaciones:

  * H₀: Primera venta (sin precio anterior de referencia)
  * H₁: Información faltante o no registrada
  * H₂: Venta al mismo precio que el valor ofertado

* **`year_build`**
  Rango de **1000 a 2024**, con media ≈ 1954. Se recomienda filtrar construcciones previas a 1800 por ser poco realistas.

* **`purchase_price`**
  Valores entre **DKK 250,000 y más de DKK 46 millones**, lo que sugiere revisar posibles *outliers* con histogramas y escala logarítmica.

* **`sqm_price`**
  Rango entre **269 y 75,000**, lo que podría indicar errores o propiedades atípicas que requieren verificación.

---

##### 3.2.1.2.2 Medidas correctivas propuestas

1. **Filtrar `year_build`** con un umbral mínimo (ej. ≥1800).
2. **Eliminar valores faltantes**, ya que son pocos y no comprometen el análisis.
3. **Detectar y eliminar duplicados** en base a `house_id` y `date`.
4. **Analizar outliers** en `purchase_price`, `sqm` y `sqm_price` con histogramas (usar log-scale si es necesario).
5. **Convertir `date` a formato legible**, derivando una nueva columna de tipo fecha a partir del timestamp original.


#### 3.2.1.3 Tratamiento de datos faltantes y duplicados

#### 3.2.1.3.1 Analisis y tratamiento de datos faltantes

#### 3.2.1.3.1.1 Identificación y tratamiento

<!-- Foto del análisis de datos faltantes -->

Los datos faltantes fueron separados para su análisis posterior, ya que representan menos del 0.1% del total de registros. Se optó por el método de análisis de casos completos, eliminando los casos con datos faltantes sin afectar significativamente el conjunto de datos.


#### 3.2.1.3.1.1 Analisis de datos faltantes

Se observó que la mayoría de los datos faltantes están asociados a un periodo de tiempo específico (quarter), lo que indica un patrón de ausencia no aleatorio. En este caso, los datos faltantes podrían clasificarse como Missing Not At Random (MNAR), ya que su presencia depende de una variable observada (el tiempo) o no observada (como cambios en el sistema de registro en ese trimestre), siguiendo la clasificación de Little y Rubin (1987).


![image-3.png](attachment:image-3.png)

*Distribución de tipos de datos presentes en las columnas.*


#### 3.2.1.3.  

![image-2.png](attachment:image-2.png)




![image-6.png](attachment:image-6.png)
*Estadísticos descriptivos, valores nulos y ceros.*

* Se identificaron algunas **inconsistencias** y registros con valores atípicos o nulos que requieren tratamiento posterior.

![image-7.png](attachment:image-7.png)

Al tratarse de una presencia menor al 0.1 %, se decide usar el método de análsis de casos completos (eliminando los casos), sin descuidar el análsis requerido para identificar la perdida de datos.

Se determina el mecanismo de perdida de datos, 
Tras inspeccionar el proceso de scrapeo en el respositorio de origen de los datos:

Se observa que la mayor perdida de datos corresponde a una de tipo parche, asociada a los primeros (~1000) IDs.
En un analsis posterior se observó una correlación positiva entre date (en formato timestap) y estos, perteneciendo todos al primer quarter registrado.

se reaizaron analisis univariados y bivariados para identificar patrones y relaciones entre variables.

![image-9.png](attachment:image-9.png)
![image-10.png](attachment:image-10.png)

![image-11.png](attachment:image-11.png)

![image-12.png](attachment:image-12.png)

Se incluyo el id para validar que los datos se encuentran ordenados y no hay duplicados.
![image-13.png](attachment:image-13.png)

Finalmente mencionar que no se encontraron registros duplicados, consecuentemente no se tomaron medidas en este aspecto.




## Modelización.  Comprende  la  aplicación  de  los  algoritmos  de  aprendizaje 
supervisado sobre la plataforma de Big Data llamada H2O y los compara.  
  
 Resultados. Comunicar los principales resultados obtenidos (uso de métricas 
y tablas comparativas).  
  
 Conclusiones. En un párrafo redactar las conclusiones del trabajo, 
especificando la técnica utilizada, los resultados obtenidos (positivos o no).  
  
 Recomendaciones. Redactar los trabajos futuros.  
  
 Referencias bibliográficas 



##  Análisis exploratorio de los datos (EDA).  


Se debe incluir la descripción  de 
las  tareas  de  inspección,  preprocesamiento,  análisis  univariado,  bivariado  y 
visualización de los datos.  
  

![Figura Y](utils/doc_src/data_analysis_flow1.png)

*Figura Y. Flujo secuencial del análisis realizado*


## Modelización.  Comprende  la  aplicación  de  los  algoritmos  de  aprendizaje 
supervisado sobre la plataforma de Big Data llamada H2O y los compara.  
  
 Resultados. Comunicar los principales resultados obtenidos (uso de métricas 
y tablas comparativas).  
  
 Conclusiones. En un párrafo redactar las conclusiones del trabajo, 
especificando la técnica utilizada, los resultados obtenidos (positivos o no).  
  
 Recomendaciones. Redactar los trabajos futuros.  
  
 Referencias bibliográficas 

### Referencias Bibliográficas

 [1] Montero, J., & Fernández-Avilés, G. (2017). La importancia de los efectos espaciales en la predicción del precio de la vivienda: una aplicación geoestadística en España. Papeles de Economía Española, 152, 102-117. https://www.funcas.es/wp-content/uploads/Migracion/Articulos/FUNCAS_PEE/152art08.pdf

[2] Larsen, K. (2020). An Assessment of the Danish Real Estate Market. MSc Thesis, Copenhagen Business School. https://research-api.cbs.dk/ws/portalfiles/portal/66775988/1043309_An_Assessment_of_the_Danish_Real_Estate_Market_.pdf

[3] Datsko, A. (2023). ANÁLISIS Y PREDICCIÓN DEL PRECIO DE LA VIVIENDA
EN MADRID UTILIZANDO TÉCNICAS DE EXPLORACIÓN DE DATOS E INTELIGENCIA ARTIFICIAL IMPLEMENTADAS EN PYTHON. Universidad Politecnica de Madrid. https://oa.upm.es/80281/1/TFG_DATSKO_ARTEM.pdf


[4] Nussupbekova, T. (2025). Denmark's Residential Property Market Analysis 2025.https://www.globalpropertyguide.com/europe/denmark/price-history

[5] Copper, A. (2021).Explaining Machine Learning Models: A Non-Technical Guide to Interpreting SHAP Analyses. Aidan Cooper. https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses


Little, RJA y Rubin, DB (2014).  Análisis estadístico con datos faltantes (Segunda edición). John Wiley & Sons.


https://stats.stackexchange.com/questions/453386/working-with-time-series-data-splitting-the-dataset-and-putting-the-model-into 


No recomiendo ningún tipo de validación cruzada (incluso la validación cruzada de series temporales es algo complicada de usar en la práctica). En su lugar, utilice una simple división entre pruebas y entrenamiento para experimentos y pruebas de concepto iniciales, etc.

Luego, al pasar a producción, no te molestes en dividir el entrenamiento, la prueba y la evaluación. Como bien señalaste, no quieres perder información valiosa de los últimos 90 días. En su lugar, en producción, entrenas varios modelos con todo el conjunto de datos y luego eliges el que te proporcione el AIC o BIC más bajo.

...
Para los métodos estadísticos, utilice una división simple de entrenamiento/prueba de series temporales para validaciones iniciales y pruebas de concepto, pero no utilice el CV para ajustar los hiperparámetros. En su lugar, entrene varios modelos en producción y utilice el AIC o el BIC como métrica para la selección automática de modelos. Además, realice este entrenamiento y selección con la mayor frecuencia posible (es decir, cada vez que obtenga nuevos datos de demanda).


Este buen hombre nos dice que usemos el AIC o el BIC

![alt text](image.png)





