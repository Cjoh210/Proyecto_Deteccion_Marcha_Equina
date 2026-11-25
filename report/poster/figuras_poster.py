"""
Figuras del póster (90 x 120 cm), dibujadas al tamaño físico de cada recuadro
de la plantilla FIDIA y con su paleta (azul 004AAD, celeste 38B6FF, gris F2F2F2).

Uso: python report/poster/figuras_poster.py   ->  report/poster/figures/es/*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qrcode
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent / "figures"
AZUL, CELESTE, GRIS_FONDO, NEGRO, GRIS = "#004AAD", "#38B6FF", "#F2F2F2", "#1F1F1F", "#9A9A9A"
CM = 1 / 2.54
plt.rcParams.update({"font.family": "Liberation Sans", "savefig.dpi": 220, "axes.edgecolor": "#BBBBBB"})

T = {
    "es": dict(
        pubs="Publicaciones por año", share="% de trabajos con visión por computador", year="Año",
        trend_title="Investigación sobre cojera, marcha y pose animal (4.024 artículos)",
        sources=["4.024 artículos\n(API OpenAlex,\nweb scraping)", "14 datasets\npúblicos", "Señales de\nsensores\n22 caballos"],
        steps=["Limpieza e\ningeniería de\ncaracterísticas", "EDA · PCA\nClustering\nReglas de asociación",
               "Ensembles\nGrid Search\nSHAP", "AWS con\nTerraform\nAPI + DynamoDB"],
        crisp="CRISP-DM",
        stat="0,71", stat_lbl="ROC-AUC con caballos que el modelo nunca vio",
        box="RANDOM FOREST OPTIMIZADO",
        note=["Validado con 5 caballos reservados.", "Acertó 7 de 9 pruebas de trote.",
              "Se apoya en la asimetría de cabeza y pelvis (SHAP)."],
        bars=["Asimetría de cabeza · anterior", "Asimetría de pelvis · posterior", "Diferencia de caderas · posterior", "Asimetría de cabeza · posterior"],
        pill="CAMBIO DE LA MEDIANA POR CABALLO CON COJERA INDUCIDA",
    ),
}


def tendencia(t, out):
    art = pd.read_csv(P / "articulos_limpio.csv")
    a = art[(art.anio >= 2000) & (art.anio <= 2025)]
    n = a.groupby("anio").size()
    v = a.groupby("anio")["tema_vision"].mean().rolling(3, center=True, min_periods=1).mean() * 100
    fig, ax = plt.subplots(figsize=(31 * CM, 15 * CM))
    ax.bar(n.index, n.values, color="#C9C9C9", width=0.75, label=t["pubs"])
    ax.set_ylabel(t["pubs"], fontsize=20, color=NEGRO)
    ax.set_xlabel(t["year"], fontsize=20)
    ax2 = ax.twinx()
    ax2.plot(v.index, v.values, color=AZUL, lw=5, marker="o", ms=8, label=t["share"])
    ax2.set_ylim(0, 70); ax2.set_ylabel(t["share"], fontsize=20, color=AZUL)
    ax2.tick_params(axis="y", colors=AZUL, labelsize=18)
    ax.tick_params(labelsize=18)
    for s in ["top"]:
        ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    ax.set_title(t["trend_title"], fontsize=22, fontweight="bold", color=NEGRO, loc="left", pad=14)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=17, frameon=False)
    fig.tight_layout(); fig.savefig(out, facecolor="white"); plt.close(fig)


def caja(ax, x, y, w, h, txt, fc, tc="white", fs=19, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.35", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def flecha(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=3, color=NEGRO, mutation_scale=28))


def metodologia(t, out):
    W, H = 40, 15
    fig = plt.figure(figsize=(W * CM, H * CM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    # fuentes (izquierda, apiladas)
    ys = [10.2, 5.7, 1.2]
    for y, s in zip(ys, t["sources"]):
        caja(ax, 0.3, y, 7.6, 3.6, s, CELESTE, tc=NEGRO, fs=16)
    # pasos
    xs, bw = [9.9, 17.5, 25.1, 32.7], 6.3
    for i, (x, s) in enumerate(zip(xs, t["steps"])):
        caja(ax, x, 4.4, bw, 6.2, s, AZUL if i < 3 else NEGRO, fs=16)
    for y in ys:
        flecha(ax, 7.95, y + 1.8, 9.95, 7.5)
    for a, b in zip(xs[:-1], xs[1:]):
        flecha(ax, a + bw + 0.1, 7.5, b - 0.1, 7.5)
    ax.text(10.0, 12.2, t["crisp"], fontsize=20, fontweight="bold", color=AZUL)
    ax.plot([10.0, 39.0], [11.9, 11.9], color=AZUL, lw=2)
    fig.savefig(out, facecolor="white"); plt.close(fig)


def resultados(t, out):
    imu = pd.read_csv(P / "imu_zancadas_limpio.csv")
    med = imu.groupby(["experimento", "id_caballo", "cojo"])[["cabeza_mindiff_mm", "sacro_mindiff_mm", "hip_hike_diff_mm"]].median().reset_index()
    def cambio(exp, var):
        m = med[med.experimento == exp].pivot(index="id_caballo", columns="cojo", values=var).dropna()
        return (m[1].median() / m[0].median() - 1) * 100
    vals = [cambio("anterior", "cabeza_mindiff_mm"), cambio("posterior", "sacro_mindiff_mm"),
            cambio("posterior", "hip_hike_diff_mm"), cambio("posterior", "cabeza_mindiff_mm")]
    W, H = 55.3, 20
    fig = plt.figure(figsize=(W * CM, H * CM), facecolor=GRIS_FONDO)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(1.2, 13.4, t["stat"], fontsize=120, fontweight="bold", color=AZUL, va="bottom")
    ax.text(1.3, 12.2, t["stat_lbl"], fontsize=21, fontweight="bold", color=NEGRO, va="center")
    ax.add_patch(FancyBboxPatch((1.2, 9.6), 17.2, 1.5, boxstyle="square,pad=0", fc=NEGRO, ec="none"))
    ax.text(1.6, 10.35, t["box"], fontsize=17, fontweight="bold", color="white", va="center")
    for i, s in enumerate(t["note"]):
        ax.text(1.3, 7.9 - i * 1.55, s, fontsize=16, color="#333333", va="center")
    ax.plot([19.8, 19.8], [1.5, 18.5], color="#BBBBBB", lw=2)
    # barras
    x0, xmax = 33.5, 52.0
    escala = (xmax - x0) / 100
    for i, (lab, v) in enumerate(zip(t["bars"], vals)):
        y = 15.6 - i * 3.3
        color = AZUL if v >= 30 else GRIS
        ax.text(x0 - 0.6, y, lab, ha="right", va="center", fontsize=18, color=NEGRO, fontweight="bold" if i == 0 else "normal")
        ax.add_patch(FancyBboxPatch((x0, y - 0.85), max(v, 1) * escala, 1.7, boxstyle="round,pad=0,rounding_size=0.4", fc=color, ec="none"))
        ax.text(x0 + max(v, 1) * escala + 0.5, y, f"+{v:.0f} %", va="center", fontsize=19, fontweight="bold", color=NEGRO)
    ax.add_patch(FancyBboxPatch((W - 30.5, 0.8), 29.5, 1.6, boxstyle="round,pad=0,rounding_size=0.8", fc=NEGRO, ec="none"))
    ax.text(W - 15.75, 1.6, t["pill"], ha="center", va="center", fontsize=15, fontweight="bold", color="white")
    fig.savefig(out, facecolor=GRIS_FONDO); plt.close(fig)


def qr(out):
    img = qrcode.make("https://github.com/Cjoh210/Proyecto_Deteccion_Marcha_Equina", border=1, box_size=20)
    img.save(out)


if __name__ == "__main__":
    for lang, t in T.items():
        d = OUT / lang; d.mkdir(parents=True, exist_ok=True)
        tendencia(t, d / "intro_tendencia.png")
        metodologia(t, d / "metodologia.png")
        resultados(t, d / "resultados.png")
        qr(d / "qr_repositorio.png")
        print("ok", lang)
