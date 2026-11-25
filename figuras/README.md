# Figuras del proyecto

Gráficas de resultados organizadas por etapa del análisis. Cada carpeta corresponde al notebook que genera sus figuras, así que al ejecutar el notebook se vuelven a crear en el mismo lugar.

## Figuras del informe

Dónde está cada figura del informe final (`report/final/CarlosBarrera_informe.pdf`):

| Informe | Archivo |
| --- | --- |
| Fig. 1 | Diagrama del sistema, dibujado dentro del informe |
| Fig. 2 | `03_imu_perfiles/imu_pca.png` |
| Fig. 3 | `02_articulos_clustering_reglas/art_kmeans_seleccion_k.png` |
| Fig. 4 | `07_cloud_aws/arquitectura_aws.png` |
| Fig. 5 | `07_cloud_aws/capturas/terraform_apply.png` |
| Fig. 6 | `07_cloud_aws/capturas/s3_openalex.png` y `athena_consulta.png` |
| Fig. 7 | `07_cloud_aws/capturas/api_respuesta.png` y `dynamodb_items.png` |
| Fig. 8 | `01_articulos_eda/art_tendencia_anual.png` |
| Fig. 9 | `02_articulos_clustering_reglas/art_tsne_clusters.png` |
| Fig. 10 | `05_catalogo_datasets/cat_matriz_aptitud.png` |
| Fig. 11 | `03_imu_perfiles/imu_pareado_caballos.png` |
| Fig. 12 | `03_imu_perfiles/imu_boxplots.png` |
| Fig. 13 | `04_ml_clasificacion/ml_roc_confusion.png` |
| Fig. 14 | `04_ml_clasificacion/ml_shap_summary.png` |
| Fig. 15 | `06_regresion/reg_comparacion.png` |
| Fig. 16 | `06_regresion/reg_lasso_coeficientes.png` |

Las demás gráficas son de apoyo y se describen abajo, carpeta por carpeta.

## 01_articulos_eda — notebook 01

| Figura | Qué muestra |
| --- | --- |
| `art_tendencia_anual.png` | Publicaciones por año y porcentaje que usa visión por computador |
| `art_distribucion_citas.png` | Distribución de las citas de los artículos |
| `art_paises_revistas.png` | Países y revistas con más publicaciones |
| `art_tipo_oa.png` | Impacto por tipo de documento y porcentaje de acceso abierto por periodo |
| `art_correlaciones.png` | Correlación entre las variables numéricas del corpus |
| `art_outliers.png` | Citas por año según la antigüedad, con los artículos atípicos marcados |

## 02_articulos_clustering_reglas — notebook 02

| Figura | Qué muestra |
| --- | --- |
| `art_svd_varianza.png` | Varianza explicada por la reducción de dimensión del texto |
| `art_kmeans_seleccion_k.png` | Selección del número de clústeres |
| `art_tsne_clusters.png` | Mapa de los perfiles temáticos encontrados |
| `art_evolucion_perfiles.png` | Evolución de cada perfil temático en el tiempo |
| `art_dendrograma.png` | Agrupamiento jerárquico, como comparación con K-Means |
| `art_reglas_scatter.png` | Reglas de asociación entre palabras clave según soporte, confianza y lift |

## 03_imu_perfiles — notebook 03

| Figura | Qué muestra |
| --- | --- |
| `imu_boxplots.png` | Métricas de simetría por zancada: sano frente a cojera inducida, anterior y posterior |
| `imu_pareado_caballos.png` | Cambio de cada caballo antes y después de la cojera inducida |
| `imu_correlaciones.png` | Correlación entre las medidas de los sensores |
| `imu_pca.png` | Proyección PCA de las zancadas |
| `imu_dendrograma_caballos.png` | Agrupamiento de los caballos según su patrón de marcha |

## 04_ml_clasificacion — notebook 04

| Figura | Qué muestra |
| --- | --- |
| `ml_roc_confusion.png` | Curva ROC y matriz de confusión con caballos de prueba |
| `ml_loho_caballos.png` | Exactitud por caballo dejando uno por fuera en cada ronda (Leave-One-Horse-Out) |
| `ml_shap_summary.png` | Variables que más influyen en la predicción (SHAP) |

## 05_catalogo_datasets — notebook 05

| Figura | Qué muestra |
| --- | --- |
| `cat_matriz_aptitud.png` | Qué datasets públicos sirven para cada necesidad del proyecto |
| `cat_dendrograma.png` | Perfiles de los datasets públicos según sus características (distancia de Gower) |

## 06_regresion — notebook 06

| Figura | Qué muestra |
| --- | --- |
| `reg_comparacion.png` | Desempeño de los modelos de regresión lineal, polinomial y Random Forest |
| `reg_lasso_coeficientes.png` | Variables que conserva Lasso |

## 07_cloud_aws

| Figura | Qué muestra |
| --- | --- |
| `arquitectura_aws.png` | Arquitectura serverless desplegada con Terraform (generada por `src/cloud/arquitectura_aws.py`) |
| `capturas/terraform_apply.png` | Despliegue de la infraestructura |
| `capturas/s3_openalex.png` | Datos de OpenAlex guardados en el data lake |
| `capturas/lambda_ingesta.png` | Ejecución de la Lambda de ingesta |
| `capturas/athena_consulta.png` | Consulta SQL sobre el data lake |
| `capturas/dynamodb_items.png` | Predicciones registradas en DynamoDB |
| `capturas/api_respuesta.png` | Respuesta de la API de inferencia |
