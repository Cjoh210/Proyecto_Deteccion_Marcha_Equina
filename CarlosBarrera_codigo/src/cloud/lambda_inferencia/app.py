"""
Lambda de inferencia (imagen Docker). Recibe las métricas de simetría de una o varias zancadas,
devuelve la probabilidad de cojera y registra la predicción en DynamoDB.

POST /predecir
{"id_caballo": "caballo_01", "zancadas": [{"cabeza_mindiff_mm": 25.1, "sacro_mindiff_mm": 9.8, ...}, ...]}
"""
import io
import json
import os
from datetime import datetime, timezone
from decimal import Decimal

import boto3
import joblib
import pandas as pd

_modelo = None


def _cargar():
    global _modelo
    if _modelo is None:  # se reutiliza entre invocaciones (contenedor caliente)
        obj = boto3.client("s3").get_object(Bucket=os.environ["BUCKET"], Key=os.environ["MODELO_KEY"])
        _modelo = joblib.load(io.BytesIO(obj["Body"].read()))
    return _modelo


def handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
        zancadas = pd.DataFrame(body["zancadas"])
        modelo = _cargar()
        cols = list(modelo.feature_names_in_)
        X = zancadas.reindex(columns=cols)  # columnas faltantes -> NaN -> imputación del pipeline
        p = modelo.predict_proba(X)[:, 1]
        p_media = float(p.mean())
        salida = {"id_caballo": body.get("id_caballo", "desconocido"), "n_zancadas": int(len(p)),
                  "probabilidad_cojera": round(p_media, 4), "cojo": p_media >= 0.5}
        boto3.resource("dynamodb").Table(os.environ["TABLA"]).put_item(Item={
            "id_caballo": salida["id_caballo"], "timestamp": datetime.now(timezone.utc).isoformat(),
            "probabilidad": Decimal(str(salida["probabilidad_cojera"])), "n_zancadas": salida["n_zancadas"]})
        return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps(salida)}
    except (KeyError, ValueError) as e:
        return {"statusCode": 400, "body": json.dumps({"error": f"Entrada inválida: {e}"})}
