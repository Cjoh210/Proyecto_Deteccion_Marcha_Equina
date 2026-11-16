# Detección de cojera equina: minería de datos, ML y nube (avance)

**Proyecto integrador · Quinto semestre · Ingeniería de Datos e IA · Universidad Santo Tomás, Tunja**
Autor: **Carlos Josué Barrera Vargas** · Asignaturas: Minería de Datos, Machine Learning y Cloud Computing
ODS: **3** (salud y bienestar, enfoque *Una Salud*) y **8** (trabajo decente y crecimiento económico: sector equino rural)

> Fase exploratoria del sistema. Antes de construir el detector de cojera por **visión por computador** (fase final),
> este avance usa minería de datos para entender el estado del arte, perfilar los datasets disponibles y validar con
> datos de sensores reales qué señales de la marcha delatan la cojera.

## Qué contiene

| Componente | Dónde | Resultado principal |
|---|---|---|
| **Web scraping** (API REST OpenAlex) | `src/scraping/openalex_scraper.py`, `src/cloud/lambda_ingesta/` | 6.472 registros crudos → 4.024 artículos limpios |
| **Limpieza + EDA** del corpus científico | `notebooks/01_articulos_limpieza_eda.ipynb` | Tendencias, outliers de impacto (IQR + z robusto) |
| **Clustering de texto + reglas de asociación** | `notebooks/02_articulos_clustering_reglas.ipynb` | 8 líneas de investigación (K-Means sobre TF-IDF/SVD), 52 reglas FP-Growth, vacío: 175 trabajos caballo+cojera+visión y solo 1 con autores de Colombia |
| **Dataset IMU de cojera inducida** (22 caballos) | `src/imu/*.py`, `notebooks/03_imu_eda_perfiles.ipynb` | 1.817 zancadas × 27 métricas de simetría; PCA, Isolation Forest, K-Means y clustering jerárquico de perfiles |
| **ML preliminar** (sano vs. cojo) | `notebooks/04_ml_clasificacion_cojera.ipynb` | 6 modelos, Grid/Random Search, validación agrupada por caballo; Random Forest: ROC-AUC 0,71 con caballos no vistos; SHAP |
| **Perfilamiento de datasets públicos** | `notebooks/05_catalogo_datasets_perfiles.ipynb` | 14 datasets, distancia de Gower + clustering jerárquico; ningún dataset público tiene video de caballos con etiqueta de cojera |
| **Infraestructura como código (AWS)** | `terraform/` | S3 data lake, Lambda + EventBridge, Glue/Athena, DynamoDB, API Gateway, ECR, CloudWatch/Budgets, réplica DR opcional |
| **Diagrama y costos** | `docs/figures/arquitectura_aws.png`, `src/cloud/estimacion_costos.py` | ≈ USD 0,13–0,38/mes (académico, prácticamente gratis) y ≈ USD 11–13,5/mes (piloto) |

## Estructura

```
├── data/
│   ├── raw/                 # respuesta de OpenAlex (CSV aplanado; el JSONL se regenera)
│   ├── processed/           # datasets limpios, resultados de clustering, reglas, ML y costos
│   └── external/            # catálogo curado de datasets (+ descarga IMU, fuera de git)
├── notebooks/               # 01–05, cada uno en .ipynb (ejecutado) y .py (formato percent)
├── src/
│   ├── scraping/            # cliente OpenAlex
│   ├── imu/                 # descarga Dataverse + extracción de características por zancada
│   ├── cloud/               # Lambdas (ingesta, inferencia + Dockerfile) y estimación de costos
│   └── py2nb.py             # convierte los .py en notebooks
├── terraform/               # IaC de AWS
├── models/                  # modelo final (.joblib) y metadatos
├── docs/figures/            # todas las figuras del informe
└── run_pipeline.sh          # ejecuta todo de principio a fin
```

## Cómo ejecutar

**Requisitos:** Python 3.11+ y, opcionalmente, Terraform ≥ 1.6 y la CLI de AWS.

```bash
git clone https://github.com/Cjoh210/Proyecto_Deteccion_Marcha_Equina.git && cd Proyecto_Deteccion_Marcha_Equina
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Opción A: reproducir los análisis con los datos incluidos (minutos)
bash run_pipeline.sh

# Opción B: todo desde cero, incluida la descarga (~6,2 GB de señales IMU)
echo "OPENALEX_API_KEY=su_clave" > .env                  # clave gratuita: https://openalex.org
bash run_pipeline.sh --descargar
```

Cada notebook también se puede abrir y ejecutar por separado con `jupyter lab notebooks/`.

### Infraestructura en AWS

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars            # completar valores (no se versiona)
terraform init
terraform plan
terraform apply                                          # crea data lake, ingesta, analítica, BD y monitoreo

# Opcional: API de inferencia
aws ecr get-login-password | docker login --username AWS --password-stdin $(terraform output -raw ecr_inferencia | cut -d/ -f1)
docker build --platform linux/amd64 --provenance=false -t $(terraform output -raw ecr_inferencia):latest ../src/cloud/lambda_inferencia
docker push $(terraform output -raw ecr_inferencia):latest
aws s3 cp ../models/modelo_cojera.joblib s3://$(terraform output -raw bucket_datalake)/models/
terraform apply -var desplegar_api_inferencia=true
curl -X POST "$(terraform output -raw url_api)" -H "Content-Type: application/json" -d @../docs/ejemplo_prediccion.json

terraform destroy                                        # al terminar, para no generar costos
```

## Fuentes de datos

- **OpenAlex** (API REST abierta, CC0): https://openalex.org
- **Smit, I. et al.** *Kinematic data of the Icelandic horse with induced lameness at walk, trot and tölt.* DataverseNL, doi:[10.34894/PEWAK6](https://doi.org/10.34894/PEWAK6), licencia CC BY 4.0. En este repositorio solo están las **características derivadas** (zancadas); las señales crudas se descargan con `src/imu/download_dataset.py`.
- Catálogo de datasets: cada fila de `data/external/catalogo_datasets.csv` incluye la URL de su fuente.

## Decisiones técnicas clave

1. **Validación agrupada por caballo** (GroupKFold, hold-out de 5 caballos y Leave-One-Horse-Out): evita que el modelo "memorice" individuos y estima el rendimiento real con caballos nuevos.
2. **Los outliers se interpretan, no se borran**: en bibliometría son los trabajos más influyentes y en el IMU son, en buena parte, la señal clínica. Solo se eliminan los artefactos físicamente imposibles (1 % de las zancadas).
3. **Pruebas estadísticas a nivel de caballo** (Wilcoxon emparejado), no de zancada, para respetar la dependencia de los datos.
4. **Arquitectura serverless** (Lambda, S3, DynamoDB on-demand): costo casi nulo cuando no hay uso, escalado automático y alta disponibilidad Multi-AZ gestionada por AWS.
