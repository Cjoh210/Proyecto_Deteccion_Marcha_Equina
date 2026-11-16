"""
Módulo de adquisición (web scraping vía API REST) - Fuente 1: corpus científico.

Descarga desde la API pública de OpenAlex los artículos relacionados con la
detección de cojera equina, el análisis de marcha y la estimación de pose animal.
Función concreta dentro del sistema: construir y actualizar periódicamente la base
de conocimiento (estado del arte) que alimenta el análisis de minería de datos.

Uso:
    export OPENALEX_API_KEY=xxxx       # clave gratuita (opcional pero recomendada)
    python src/scraping/openalex_scraper.py --max-per-query 1500

Salida:
    data/raw/openalex_works_raw.jsonl   (respuesta cruda, un JSON por línea)
    data/raw/openalex_works_raw.csv     (tabla aplanada, sin limpiar)
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import requests

BASE_URL = "https://api.openalex.org/works"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

# Consultas temáticas (búsqueda en título + resumen). Cada artículo conserva la
# consulta que lo trajo, lo que permite analizar el solapamiento entre temas.
QUERIES = {
    "cojera_equina": '(horse OR equine OR equid) AND (lameness OR lame)',
    "marcha_equina": '(horse OR equine) AND (gait OR locomotion OR kinematics) AND (analysis OR asymmetry)',
    "sensores_inerciales": '(horse OR equine) AND (accelerometer OR "inertial sensor" OR IMU OR gyroscope)',
    "vision_pose_animal": '("pose estimation" OR keypoint OR markerless OR "computer vision") AND (horse OR equine OR animal OR quadruped)',
    "cojera_ml_otras_especies": '(lameness) AND ("machine learning" OR "deep learning" OR "neural network" OR classification)',
}

SELECT_FIELDS = ",".join([
    "id", "doi", "display_name", "publication_year", "publication_date", "type",
    "language", "cited_by_count", "fwci", "is_retracted", "open_access",
    "primary_location", "authorships", "keywords", "concepts", "primary_topic",
    "abstract_inverted_index", "referenced_works_count", "countries_distinct_count",
])


def get_session():
    s = requests.Session()
    key = os.getenv("OPENALEX_API_KEY")
    if key:
        s.headers["Authorization"] = f"Bearer {key}"
    s.headers["User-Agent"] = "ProyectoIntegrador-USTA/1.0 (investigacion academica)"
    return s


def fetch_query(session, name, query, max_records, per_page=200):
    """Paginación por cursor; respeta límites con reintentos exponenciales."""
    cursor, got = "*", 0
    while cursor and got < max_records:
        params = {
            "filter": f"title_and_abstract.search:{query}",
            "per-page": per_page,
            "cursor": cursor,
            "select": SELECT_FIELDS,
        }
        for attempt in range(5):
            r = session.get(BASE_URL, params=params, timeout=60)
            if r.status_code == 200:
                break
            wait = 2 ** attempt
            print(f"  [{name}] HTTP {r.status_code}, reintento en {wait}s")
            time.sleep(wait)
        else:
            raise RuntimeError(f"Fallo persistente en la consulta {name}: {r.text[:200]}")
        payload = r.json()
        results = payload.get("results", [])
        if not results:
            break
        for w in results:
            w["_query"] = name
            yield w
        got += len(results)
        cursor = payload["meta"].get("next_cursor")
        print(f"  [{name}] {got} / {payload['meta']['count']}")
        time.sleep(0.2)  # cortesía con la API


def rebuild_abstract(inv):
    """OpenAlex entrega el resumen como índice invertido {palabra: [posiciones]}."""
    if not inv:
        return None
    pos = [(i, w) for w, idx in inv.items() for i in idx]
    return " ".join(w for _, w in sorted(pos))


def flatten(w):
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    oa = w.get("open_access") or {}
    auths = w.get("authorships") or []
    countries = sorted({c for a in auths for c in (a.get("countries") or [])})
    institutions = sorted({i.get("display_name") for a in auths for i in (a.get("institutions") or []) if i.get("display_name")})
    topic = w.get("primary_topic") or {}
    return {
        "openalex_id": w.get("id"),
        "doi": w.get("doi"),
        "titulo": w.get("display_name"),
        "anio": w.get("publication_year"),
        "fecha": w.get("publication_date"),
        "tipo": w.get("type"),
        "idioma": w.get("language"),
        "citas": w.get("cited_by_count"),
        "fwci": w.get("fwci"),
        "retractado": w.get("is_retracted"),
        "acceso_abierto": oa.get("is_oa"),
        "estado_oa": oa.get("oa_status"),
        "revista": src.get("display_name"),
        "tipo_fuente": src.get("type"),
        "n_autores": len(auths),
        "paises": ";".join(countries),
        "n_paises": w.get("countries_distinct_count"),
        "instituciones": ";".join(institutions[:10]),
        "n_referencias": w.get("referenced_works_count"),
        "keywords": ";".join(k.get("display_name", "") for k in (w.get("keywords") or [])),
        "conceptos": ";".join(c["display_name"] for c in (w.get("concepts") or []) if c.get("score", 0) >= 0.3),
        "topico": topic.get("display_name"),
        "subcampo": (topic.get("subfield") or {}).get("display_name"),
        "campo": (topic.get("field") or {}).get("display_name"),
        "resumen": rebuild_abstract(w.get("abstract_inverted_index")),
        "consulta": w.get("_query"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-query", type=int, default=1500)
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = get_session()
    jsonl = RAW_DIR / "openalex_works_raw.jsonl"
    rows = []
    with open(jsonl, "w", encoding="utf-8") as fj:
        for name, q in QUERIES.items():
            print(f"Consulta: {name}")
            for w in fetch_query(session, name, q, args.max_per_query):
                fj.write(json.dumps(w, ensure_ascii=False) + "\n")
                rows.append(flatten(w))
    out = RAW_DIR / "openalex_works_raw.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"Registros crudos (con duplicados entre consultas): {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
