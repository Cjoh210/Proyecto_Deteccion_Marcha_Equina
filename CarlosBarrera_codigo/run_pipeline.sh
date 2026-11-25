#!/usr/bin/env bash
# Ejecuta el pipeline completo de principio a fin (reproducibilidad).
# Uso:  bash run_pipeline.sh            -> usa los datos ya incluidos en data/
#       bash run_pipeline.sh --descargar -> vuelve a descargar OpenAlex y el dataset IMU (~6.2 GB)
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--descargar" ]]; then
  echo "== 1. Scraping OpenAlex =="
  [[ -f .env ]] && export $(grep -v '^#' .env | xargs)
  python src/scraping/openalex_scraper.py --max-per-query 1500
  echo "== 2. Descarga dataset IMU (pruebas a mano) =="
  python src/imu/download_dataset.py --dest data/external/imu_cojera
  echo "== 3. Extracción de características por zancada =="
  python src/imu/feature_extraction.py --data-dir data/external/imu_cojera --out data/processed/imu_zancadas.csv
fi

echo "== 4. Notebooks (limpieza, EDA, clustering, reglas, ML, catálogo) =="
cd notebooks
for nb in 01_articulos_limpieza_eda 02_articulos_clustering_reglas 03_imu_eda_perfiles 04_ml_clasificacion_cojera 05_catalogo_datasets_perfiles 06_regresion_impacto_articulos; do
  echo "   -> $nb"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb.ipynb"
done
cd ..

echo "== 5. Estimación de costos AWS =="
python src/cloud/estimacion_costos.py

echo "== 6. Validación de Terraform =="
if command -v terraform >/dev/null; then
  (cd terraform && terraform init -backend=false -input=false >/dev/null && terraform validate)
else
  echo "   (terraform no instalado: se omite)"
fi
echo "Listo."
