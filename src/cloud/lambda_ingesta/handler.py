"""
Lambda de ingesta (web scraping vía API REST de OpenAlex).
Se ejecuta semanalmente (EventBridge) y guarda el JSON crudo en
s3://$BUCKET/raw/openalex/fecha=YYYY-MM-DD/<consulta>.jsonl

Solo usa la biblioteca estándar + boto3 (incluido en el runtime de Lambda),
así el paquete no necesita dependencias externas.
"""
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import date

import boto3

QUERIES = {
    "cojera_equina": '(horse OR equine OR equid) AND (lameness OR lame)',
    "marcha_equina": '(horse OR equine) AND (gait OR locomotion OR kinematics) AND (analysis OR asymmetry)',
    "sensores_inerciales": '(horse OR equine) AND (accelerometer OR "inertial sensor" OR IMU OR gyroscope)',
    "vision_pose_animal": '("pose estimation" OR keypoint OR markerless OR "computer vision") AND (horse OR equine OR animal OR quadruped)',
    "cojera_ml_otras_especies": '(lameness) AND ("machine learning" OR "deep learning" OR "neural network" OR classification)',
}
SELECT = ("id,doi,display_name,publication_year,publication_date,type,language,cited_by_count,fwci,is_retracted,"
          "open_access,primary_location,authorships,keywords,concepts,primary_topic,abstract_inverted_index,"
          "referenced_works_count,countries_distinct_count")

s3 = boto3.client("s3")


def _api_key():
    nombre = os.environ.get("PARAM_CLAVE")
    if not nombre:
        return None
    try:
        v = boto3.client("ssm").get_parameter(Name=nombre, WithDecryption=True)["Parameter"]["Value"]
        return None if v == "SIN_CLAVE" else v
    except Exception:
        return None  # sin clave se usa el cupo público compartido


def _get(url, key):
    req = urllib.request.Request(url, headers={"User-Agent": "ProyectoIntegrador-USTA/1.0"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    for intento in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            time.sleep(2 ** intento)
            ultimo = e
    raise RuntimeError(f"Fallo persistente: {ultimo}")


def lambda_handler(event, context):
    bucket, prefijo = os.environ["BUCKET"], os.environ.get("PREFIJO", "raw/openalex/")
    max_q = int(os.environ.get("MAX_POR_CONSULTA", "1500"))
    key, hoy, resumen = _api_key(), date.today().isoformat(), {}
    for nombre, q in QUERIES.items():
        # Se escribe página por página en /tmp (almacenamiento efímero de Lambda) para no acumular
        # miles de registros en memoria; luego se sube el archivo completo a S3.
        ruta, cursor, n = f"/tmp/{nombre}.jsonl", "*", 0
        with open(ruta, "w", encoding="utf-8") as f:
            while cursor and n < max_q:
                params = urllib.parse.urlencode({"filter": f"title_and_abstract.search:{q}", "per-page": 200,
                                                 "cursor": cursor, "select": SELECT})
                data = _get(f"https://api.openalex.org/works?{params}", key)
                res = data.get("results", [])
                if not res:
                    break
                for w in res:
                    w["_query"] = nombre
                    f.write(json.dumps(w, ensure_ascii=False) + "\n")
                n += len(res)
                cursor = data["meta"].get("next_cursor")
                del data, res
        s3.upload_file(ruta, bucket, f"{prefijo}fecha={hoy}/{nombre}.jsonl")
        os.remove(ruta)
        resumen[nombre] = n
    print(json.dumps({"fecha": hoy, "registros": resumen}))
    return {"statusCode": 200, "registros": resumen}
