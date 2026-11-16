"""
Extracción de características de marcha por zancada a partir de las señales IMU
(dataset: Smit et al., "Kinematic data of the Icelandic horse with induced lameness",
DataverseNL, doi:10.34894/PEWAK6, CC-BY-4.0).

Para cada archivo de pruebas a mano (HAND) se toman los segmentos de TROTE en LÍNEA RECTA
(la condición estándar de la evaluación clínica de cojera) y, por cada zancada, se calculan
métricas clásicas de simetría de la marcha en cabeza, cruz (withers), sacro y tuberosidades
coxales izquierda/derecha:

  * Desplazamiento vertical (doble integración de la aceleración vertical, con filtros paso-alto):
      - MinDiff: diferencia absoluta entre los dos mínimos de la zancada (uno por diagonal)
      - MaxDiff: diferencia absoluta entre los dos máximos
      - ROM: rango de movimiento vertical
  * Harmonic Ratio (HR) de la aceleración vertical: armónicos pares / impares (simetría temporal)
  * Hip hike: diferencia de ROM entre tuberosidad coxal izquierda y derecha
  * Duración de la zancada

Uso:
    python src/imu/feature_extraction.py --data-dir <carpeta_descarga> --out data/processed/imu_zancadas.csv
"""
import argparse
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, decimate, find_peaks, sosfiltfilt

FS_RAW, DEC = 500, 5
FS = FS_RAW // DEC  # 100 Hz tras decimar
NODOS = {"cabeza": "hd", "cruz": "wth", "sacro": ("sacrum", "sac"), "tc_izq": "LTC", "tc_der": "RTC"}
ACC_COLS = slice(6, 9)  # ax, ay, az (m/s^2) según h.<nodo>.columns


def _sos(kind, f):
    return butter(4, f, btype=kind, fs=FS, output="sos")


SOS_BP = _sos("bandpass", [0.3, 20])
SOS_HP = _sos("highpass", 0.5)


def load_imu(path):
    """Devuelve {nodo: matriz (N x 38)}. Los archivos vienen en dos formatos de MATLAB:
    v5 (scipy.io) y v7.3 (HDF5, se lee con h5py y se transpone)."""
    with open(path, "rb") as f:
        v73 = b"MATLAB 7.3" in f.read(128)
    if v73:
        import h5py
        with h5py.File(path, "r") as f:
            return {k: f["h"][k]["data"][()].T for k in f["h"].keys() if "data" in f["h"][k]}
    h = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["h"]
    return {k: getattr(h, k).data for k in h._fieldnames}


def _get_node(h, name):
    names = name if isinstance(name, tuple) else (name,)
    for n in names:
        if n in h:
            return h[n]
    return None


def vertical_acc(acc):
    """Proyecta la aceleración sobre la dirección media de la gravedad (aprox. válida en trote recto)."""
    acc = pd.DataFrame(acc).interpolate(limit_direction="both").to_numpy()
    if np.isnan(acc).any():
        return None
    g = acc.mean(axis=0)
    av = acc @ (g / np.linalg.norm(g)) - np.linalg.norm(g)
    av = decimate(av, DEC, zero_phase=True)
    return sosfiltfilt(SOS_BP, av)


def displacement(av):
    v = sosfiltfilt(SOS_HP, cumulative_trapezoid(av, dx=1 / FS, initial=0))
    d = sosfiltfilt(SOS_HP, cumulative_trapezoid(v, dx=1 / FS, initial=0))
    return d * 1000  # mm


def stride_period(av):
    """Periodo de zancada por autocorrelación (pico entre 0.5 y 1.0 s en trote)."""
    x = av - av.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    lo, hi = int(0.5 * FS), int(1.0 * FS)
    return (lo + np.argmax(ac[lo:hi])) / FS


def harmonic_ratio(seg, n_harm=20):
    amp = np.abs(np.fft.rfft(seg - seg.mean()))[1:n_harm + 1]
    pares, impares = amp[1::2].sum(), amp[0::2].sum()  # armónico 1 = índice 0 (impar)
    return pares / impares if impares > 0 else np.nan


def process_file(args):
    imu_path, timing_path, meta = args
    try:
        h = load_imu(imu_path)
        tm = sio.loadmat(timing_path, squeeze_me=True, struct_as_record=False)
        segs = np.atleast_1d(tm.get("segments", tm.get("result_mat")))
    except Exception as e:  # archivo corrupto o formato inesperado
        print("ERROR", imu_path, e)
        return []
    rows = []
    for si, s in enumerate(segs):
        if str(s.gait_class) != "trot" or str(s.direction) != "straight":
            continue
        a, b = int(s.start), int(s.stop)
        if b - a < 3 * FS_RAW:  # segmentos de menos de 3 s no tienen zancadas estables
            continue
        sig = {}
        for k, n in NODOS.items():
            d = _get_node(h, n)
            if d is None or d.shape[0] < b:
                continue
            seg = d[a:b, ACC_COLS]
            if np.isnan(seg).mean() > 0.2:
                continue
            av = vertical_acc(seg)
            if av is not None:
                sig[k] = (av, displacement(av))
        if "sacro" not in sig:
            continue
        av_s, d_s = sig["sacro"]
        T = stride_period(av_s)
        step = T / 2
        # máximos del sacro (fase de suspensión) delimitan los pasos; mínimos para los máximos de cada paso
        maxs, _ = find_peaks(d_s, distance=int(0.6 * step * FS))
        mins, _ = find_peaks(-d_s, distance=int(0.6 * step * FS))
        margin = int(0.5 * FS)  # descarta bordes afectados por el filtrado
        maxs = maxs[(maxs > margin) & (maxs < len(d_s) - margin)]
        for zi in range(0, len(maxs) - 2, 2):
            p0, p1, p2 = maxs[zi], maxs[zi + 1], maxs[zi + 2]
            dur = (p2 - p0) / FS
            row = {**meta, "segmento": si, "zancada": zi // 2, "duracion_zancada_s": dur,
                   "periodo_autocorr_s": T, "n_strides_segmento": int(s.n_strides)}
            m_in = mins[(mins > p0) & (mins < p2)]
            for k, (av, d) in sig.items():
                s1, s2 = d[p0:p1], d[p1:p2]
                if len(s1) < 5 or len(s2) < 5:
                    continue
                row[f"{k}_mindiff_mm"] = abs(s1.min() - s2.min())
                row[f"{k}_rom_mm"] = d[p0:p2].max() - d[p0:p2].min()
                if len(m_in) >= 2:  # máximos por paso: ventanas centradas en los mínimos del sacro
                    c = (m_in[0] + m_in[1]) // 2
                    w1, w2 = d[p0:c], d[c:p2]
                    row[f"{k}_maxdiff_mm"] = abs(w1.max() - w2.max()) if len(w1) > 3 and len(w2) > 3 else np.nan
                row[f"{k}_hr"] = harmonic_ratio(av[p0:p2])
                row[f"{k}_acc_rms"] = float(np.sqrt(np.mean(av[p0:p2] ** 2)))
            if "tc_izq_rom_mm" in row and "tc_der_rom_mm" in row:
                row["hip_hike_diff_mm"] = abs(row["tc_izq_rom_mm"] - row["tc_der_rom_mm"])
            rows.append(row)
    return rows


def build_jobs(data_dir):
    jobs = []
    for imu in sorted(glob.glob(os.path.join(data_dir, "**", "IMU_*", "*_HAND*.mat"), recursive=True)):
        base = os.path.basename(imu)
        parent = os.path.dirname(os.path.dirname(imu))
        timing = os.path.join(parent, "Segment_timings", base)
        if not os.path.exists(timing):
            continue
        m = re.match(r"Horse([A-L])-(BAS|IND)_HAND(?:_(LH|RH))?\.mat", base)
        if not m:
            continue
        letra, cond, limb = m.groups()
        exp = "anterior" if "Forelimb" in imu else "posterior"
        meta = {"experimento": exp, "caballo": letra, "id_caballo": f"{exp[:3]}_{letra}",
                "condicion": cond, "cojo": int(cond == "IND"),
                "miembro_inducido": "LF" if exp == "anterior" else limb, "archivo": base}
        jobs.append((imu, timing, meta))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="data/processed/imu_zancadas.csv")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = build_jobs(args.data_dir)
    print(f"Archivos a procesar: {len(jobs)}")
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(process_file, jobs):
            rows.extend(r)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Zancadas extraídas: {len(df)} -> {args.out}")


if __name__ == "__main__":
    main()
