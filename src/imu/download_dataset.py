"""
Descarga reproducible del dataset IMU de cojera inducida (DataverseNL, doi:10.34894/PEWAK6, CC-BY-4.0)
usando la API nativa de Dataverse. Por defecto solo baja las pruebas "a mano" (HAND, ~6.2 GB).

Uso:
    python src/imu/download_dataset.py --dest data/external/imu_cojera            # solo HAND
    python src/imu/download_dataset.py --dest data/external/imu_cojera --all      # todo (~19 GB)
"""
import argparse
import os

import requests

API = "https://dataverse.nl/api"
DOI = "doi:10.34894/PEWAK6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/external/imu_cojera")
    ap.add_argument("--all", action="store_true", help="incluye pruebas montadas (RIDER)")
    args = ap.parse_args()
    meta = requests.get(f"{API}/datasets/:persistentId/", params={"persistentId": DOI}, timeout=60).json()
    files = meta["data"]["latestVersion"]["files"]
    sel = [f for f in files if args.all or "_HAND" in f["dataFile"]["filename"] or f["dataFile"]["filename"] == "read_me.txt"]
    total = sum(f["dataFile"].get("filesize", 0) for f in sel) / 1e9
    print(f"Archivos: {len(sel)} | Tamaño: {total:.2f} GB")
    for i, f in enumerate(sel, 1):
        df = f["dataFile"]
        out = os.path.join(args.dest, f.get("directoryLabel", ""), df["filename"])
        if os.path.exists(out) and os.path.getsize(out) == df.get("filesize"):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with requests.get(f"{API}/access/datafile/{df['id']}", stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(out, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        print(f"[{i}/{len(sel)}] {out}")


if __name__ == "__main__":
    main()
