# %% [markdown]
# # 05 · Catálogo de datasets públicos
# Entrada: `data/external/catalogo_datasets.csv` (14 datasets; los campos sin dato quedan vacíos).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG = ROOT / "figuras" / "05_catalogo_datasets"; OUT = ROOT / "data" / "processed"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})

cat = pd.read_csv(ROOT / "data" / "external" / "catalogo_datasets.csv")
print("Datasets:", len(cat))
cat[["nombre", "anio", "modalidad", "especies", "n_muestras", "unidad_muestra", "etiqueta_cojera", "licencia"]]

# %% [markdown]
# ## 1. Faltantes y transformación

# %%
print((cat.isna().mean() * 100).round(0)[lambda s: s > 0])
cat["modalidad_grupo"] = np.select(
    [cat.modalidad.str.contains("IMU"), cat.modalidad.str.contains("mocap"), cat.modalidad.str.contains("video"),
     cat.modalidad.str.contains("imagen")],
    ["Sensores IMU", "Video + mocap 3D", "Video", "Imagen"], "Otro")
cat["log_muestras"] = np.log10(cat["n_muestras"])
cat["n_keypoints"] = cat["n_keypoints"].fillna(-1)  # -1 = no reportado (se excluye en la distancia)
cat["apertura"] = np.select([cat.uso_comercial_confirmado == 1, cat.acceso.str.startswith("abierto")], [2, 1], 0)

# %%
def gower(df, num, binar):
    n = len(df); D = np.zeros((n, n)); W = np.zeros((n, n))
    for c in num:
        x = df[c].to_numpy(float); rng = np.nanmax(x) - np.nanmin(x)
        for i in range(n):
            ok = ~np.isnan(x) & ~np.isnan(x[i])
            D[i, ok] += np.abs(x[i] - x[ok]) / rng; W[i, ok] += 1
    for c in binar:
        x = df[c].to_numpy(float)
        D += (x[:, None] != x[None, :]); W += 1
    return D / W

dfg = cat.copy()
dfg["kp"] = dfg["n_keypoints"].replace(-1, np.nan)
for m in ["Sensores IMU", "Video + mocap 3D", "Video", "Imagen"]:
    dfg[f"mod_{m}"] = (dfg.modalidad_grupo == m).astype(int)
NUM = ["anio", "log_muestras", "kp", "apertura"]
BIN = ["incluye_caballo", "solo_caballo", "etiqueta_cojera", "etiqueta_comportamiento", "video_temporal", "datos_3d"] + \
      [c for c in dfg if c.startswith("mod_")]
D = gower(dfg, NUM, BIN)

# %% [markdown]
# ## 2. Clustering jerárquico (enlace promedio sobre Gower)

# %%
Z = linkage(squareform(D, checks=False), method="average")
plt.figure(figsize=(11, 4.8))
dendrogram(Z, labels=cat["nombre"].tolist(), orientation="right", color_threshold=0.7 * Z[:, 2].max(), leaf_font_size=9)
plt.title("Perfiles de datasets públicos (distancia de Gower, enlace promedio)"); plt.xlabel("Distancia")
plt.savefig(FIG / "cat_dendrograma.png"); plt.show()

cat["perfil"] = fcluster(Z, 4, criterion="maxclust")
resumen = cat.groupby("perfil").agg(n=("nombre", "size"), datasets=("nombre", lambda s: " | ".join(s)),
                                   pct_caballo=("incluye_caballo", "mean"), pct_cojera=("etiqueta_cojera", "mean"),
                                   pct_video=("video_temporal", "mean"), mediana_muestras=("n_muestras", "median"))
pd.set_option("display.max_colwidth", 200)
resumen

# %% [markdown]
# ## 3. Matriz de aptitud
# Puntaje de 0 a 1 por criterio; la etiqueta de cojera y la especie pesan más.

# %%
criterios = pd.DataFrame({k: v.to_numpy() for k, v in {
    "Es caballo": cat["solo_caballo"] * 1.0 + (cat["incluye_caballo"] - cat["solo_caballo"]) * 0.5,
    "Etiqueta de cojera": cat["etiqueta_cojera"].astype(float),
    "Video / temporal": cat["video_temporal"].astype(float),
    "Keypoints anatómicos": (cat["n_keypoints"] > 0).astype(float),
    "Escala (log muestras)": ((cat["log_muestras"] - cat["log_muestras"].min()) /
                              (cat["log_muestras"].max() - cat["log_muestras"].min())).fillna(0),
    "Licencia / acceso": cat["apertura"] / 2,
}.items()}, index=cat["nombre"])
pesos = pd.Series({"Es caballo": .25, "Etiqueta de cojera": .25, "Video / temporal": .15,
                   "Keypoints anatómicos": .15, "Escala (log muestras)": .1, "Licencia / acceso": .1})
criterios["Puntaje"] = criterios[pesos.index].mul(pesos).sum(axis=1)
criterios = criterios.sort_values("Puntaje", ascending=False)
plt.figure(figsize=(9, 6.2))
sns.heatmap(criterios, annot=True, fmt=".2f", cmap="YlGn", cbar=False, linewidths=.5)
plt.title("Aptitud de cada dataset para detectar cojera equina por visión"); plt.ylabel("")
plt.savefig(FIG / "cat_matriz_aptitud.png"); plt.show()

# %% [markdown]
# ## 4. Conteo del vacío

# %%
vacio = pd.Series({
    "Datasets con caballos": int(cat.incluye_caballo.sum()),
    "…con etiqueta de cojera": int((cat.incluye_caballo & cat.etiqueta_cojera).sum()),
    "…con cojera + video público": int((cat.incluye_caballo & cat.etiqueta_cojera & cat.video_temporal &
                                        cat.modalidad_grupo.isin(["Video", "Video + mocap 3D"]) & cat.acceso.str.startswith("abierto")).sum()),
    "…con cojera y licencia comercial": int((cat.incluye_caballo & cat.etiqueta_cojera & (cat.uso_comercial_confirmado == 1)).sum()),
})
vacio.to_frame("n")

# %%
cat.to_csv(OUT / "catalogo_datasets_perfilado.csv", index=False)
criterios.to_csv(OUT / "catalogo_aptitud.csv")
print("Exportado.")
