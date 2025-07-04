#  Análisis exploratorio y modelado predictivo de precios residenciales en Dinamarca (1992–2024)

Kaggle datset: [Danish Residential Housing Prices 1992-2024](https://www.kaggle.com/datasets/martinfrederiksen/danish-residential-housing-prices-1992-2024/data) 


## 1. Descripción del caso de uso

El precio de la vivienda es un tema socioeconómico de gran relevancia, pues la compra de una casa suele ser la inversión más importante en la vida de una familia[1]. En Dinamarca, al igual que en muchos países, el mercado inmobiliario ha experimentado notables alzas de precios en las últimas décadas. Por ejemplo, el precio promedio de una vivienda unifamiliar aumentó **153,9% en términos reales** entre 1992 y 2020[2]. Estas fluctuaciones incluyen periodos de **boom** seguidos de ajustes: antes de la crisis financiera de 2008 los precios crecieron aceleradamente y luego cayeron \~20% para 2009, recuperándose en la década siguiente[2]. Este comportamiento cíclico ha generado interrogantes sobre la existencia de burbujas inmobiliarias y la sostenibilidad de los precios respecto a fundamentos económicos[2].

Dada la importancia de la vivienda tanto para la economía nacional como para el bienestar social, **analizar y predecir los precios de las viviendas** resulta fundamental. Una predicción precisa ayuda a compradores y vendedores a tomar decisiones informadas, a la vez que permite a planificadores y entidades financieras evaluar riesgos. Modelos de *machine learning* ya han mostrado eficacia en la predicción de precios inmobiliarios[4], pudiendo descubrir patrones ocultos en los datos históricos. En este proyecto, proponemos aplicar técnicas de Big Data y aprendizaje supervisado para **encontrar patrones y predecir el precio de viviendas residenciales en Dinamarca** usando datos históricos de 1992 a 2024. Se busca no solo alta precisión predictiva sino también **interpretabilidad**, de modo que los resultados brinden conocimiento claro sobre **qué factores influyen** en el precio (por ejemplo, ubicación, tamaño, antigüedad, etc.). Esto está alineado con la tendencia hacia **IA explicable**, utilizando métodos como SHAP o LIME para interpretar modelos complejos[5]. En resumen, el caso de uso se enfoca en demostrar cómo el análisis de grandes volúmenes de datos inmobiliarios puede apoyar la toma de decisiones en el mercado de la vivienda, un ámbito de gran impacto económico y social.



### Referencias Bibligraficas

 [1] Montero, J., & Fernández-Avilés, G. (2017). La importancia de los efectos espaciales en la predicción del precio de la vivienda: una aplicación geoestadística en España. Papeles de Economía Española, 152, 102-117. https://www.funcas.es/wp-content/uploads/Migracion/Articulos/FUNCAS_PEE/152art08.pdf

[2] Larsen, K. (2020). An Assessment of the Danish Real Estate Market. MSc Thesis, Copenhagen Business School. https://research-api.cbs.dk/ws/portalfiles/portal/66775988/1043309_An_Assessment_of_the_Danish_Real_Estate_Market_.pdf

[3] Datsko, A. (2023). ANÁLISIS Y PREDICCIÓN DEL PRECIO DE LA VIVIENDA
EN MADRID UTILIZANDO TÉCNICAS DE EXPLORACIÓN DE DATOS E INTELIGENCIA ARTIFICIAL IMPLEMENTADAS EN PYTHON. Universidad Politecnica de Madrid. https://oa.upm.es/80281/1/TFG_DATSKO_ARTEM.pdf


[4] Nussupbekova, T. (2025). Denmark's Residential Property Market Analysis 2025.https://www.globalpropertyguide.com/europe/denmark/price-history

[5] Copper, A. (2021).Explaining Machine Learning Models: A Non-Technical Guide to Interpreting SHAP Analyses. Aidan Cooper. https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses