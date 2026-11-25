# %% [markdown]
# # 03 · Dataset IMU de cojera inducida: EDA, outliers, PCA y perfiles (clustering)
# **Fuente:** Smit et al., *Kinematic data of the Icelandic horse with induced lameness at walk, trot and tölt*,
# DataverseNL, doi:10.34894/PEWAK6 (CC-BY-4.0). 22 caballos (10 con cojera anterior inducida y 12 con cojera
# posterior inducida), medidos antes (BAS) y durante (IND) la inducción, con 16 sensores IMU a 500 Hz.
#
# **Preparación (script `src/imu/feature_extraction.py`):** se toman las pruebas a mano, trote en línea recta, y se calculan
# por zancada métricas de simetría vertical de cabeza, cruz, sacro y tuberosidades coxales. Esta es la ingeniería
# de características: se pasa de ~50 millones de muestras crudas a ~1.800 zancadas × 30 variables con significado clínico.

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG = ROOT / "figuras" / "03_imu_perfiles"; OUT = ROOT / "data" / "processed"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})
SEED = 42

df = pd.read_csv(OUT / "imu_zancadas.csv")
df["estado"] = df["cojo"].map({0: "Sano (BAS)", 1: "Cojo (IND)"})
FEATS = [c for c in df.columns if c.endswith(("_mm", "_hr", "_rms")) or c in ["duracion_zancada_s"]]
print(df.shape, "| caballos:", df.id_caballo.nunique(), "| variables:", len(FEATS))
df.groupby(["experimento", "estado"]).agg(zancadas=("zancada", "size"), caballos=("id_caballo", "nunique"))

# %% [markdown]
# ## 1. Calidad: nulos y valores físicamente imposibles

# %%
nul = df[FEATS].isna().mean().mul(100).round(2)
print("Variables con nulos (%):\n", nul[nul > 0])
print("\nArchivos sin sensor de cruz:", df.loc[df["cruz_rom_mm"].isna(), "archivo"].unique())

# %% [markdown]
# **Nulos:** solo las variables de la cruz tienen faltantes (2 %): en una prueba el sensor de la cruz no registró.
# Es un faltante **por archivo** (no depende del valor), así que se imputa de forma jerárquica: primero con la
# **mediana del mismo caballo y condición**, que conserva su línea base individual; si el caballo no tiene ese sensor
# en toda la prueba (caso `ant_G`, basal), se usa la mediana de su experimento y condición. En el modelo de ML la imputación va dentro del *pipeline*
# para no filtrar información del conjunto de prueba.
#
# **Valores imposibles (artefactos):** una zancada de trote dura ~0,5–0,8 s y el desplazamiento vertical del tronco no
# supera ~150 mm. Valores fuera de esos rangos vienen de transiciones de marcha, tropiezos o fallos de integración.

# %%
for c in [c for c in FEATS if c.startswith("cruz_")]:
    df[c] = df[c].fillna(df.groupby(["id_caballo", "condicion"])[c].transform("median"))
    df[c] = df[c].fillna(df.groupby(["experimento", "condicion"])[c].transform("median"))  # respaldo
print("Nulos restantes:", int(df[FEATS].isna().sum().sum()))
artefacto = (~df["duracion_zancada_s"].between(0.45, 0.85)) | (df[[c for c in FEATS if c.endswith("rom_mm")]] > 200).any(axis=1)
print(f"Zancadas artefacto: {artefacto.sum()} ({artefacto.mean()*100:.1f} %)")
df_art = df[artefacto]
df = df[~artefacto].reset_index(drop=True)
print("Zancadas válidas:", len(df))

# %% [markdown]
# ## 2. Estadística descriptiva por grupo

# %%
clave = ["cabeza_mindiff_mm", "cabeza_maxdiff_mm", "sacro_mindiff_mm", "sacro_maxdiff_mm", "hip_hike_diff_mm",
         "cabeza_hr", "sacro_hr", "duracion_zancada_s"]
df.groupby(["experimento", "estado"])[clave].median().T.round(2)

# %% [markdown]
# ### Pruebas estadísticas (por caballo, emparejadas)
# Las zancadas de un mismo caballo no son independientes. Por eso la prueba correcta se hace a nivel de **caballo**:
# mediana por caballo y condición, y **Wilcoxon de rangos con signo** (emparejado y no paramétrico).

# %%
med = df.groupby(["experimento", "id_caballo", "cojo"])[clave].median().reset_index()
res = []
for exp in ["anterior", "posterior"]:
    m = med[med.experimento == exp].pivot(index="id_caballo", columns="cojo")
    for c in clave:
        par = m[c].dropna()
        if len(par) >= 5:
            w = stats.wilcoxon(par[1], par[0])
            res.append({"experimento": exp, "variable": c, "n_caballos": len(par),
                        "mediana_BAS": par[0].median(), "mediana_IND": par[1].median(),
                        "cambio_%": (par[1].median() / par[0].median() - 1) * 100, "p_valor": w.pvalue})
tests = pd.DataFrame(res)
tests["significativo_0.05"] = tests["p_valor"] < 0.05
tests.round(3)

# %% [markdown]
# ## 3. Visualizaciones

# %%
fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
for ax, c in zip(axes.ravel(), clave):
    sns.boxplot(data=df, x="experimento", y=c, hue="estado", ax=ax, fliersize=1.5,
                palette={"Sano (BAS)": "#4C72B0", "Cojo (IND)": "#C44E52"})
    ax.set_title(c); ax.set_xlabel(""); ax.set_ylabel("")
    if ax is not axes[0, 0]:
        ax.get_legend().remove()
plt.suptitle("Métricas de simetría por zancada: sano vs. cojera inducida", y=1.01)
plt.tight_layout(); plt.savefig(FIG / "imu_boxplots.png"); plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, (exp, var, tit) in zip(axes, [("anterior", "cabeza_mindiff_mm", "Cojera anterior → asimetría de cabeza"),
                                      ("posterior", "sacro_mindiff_mm", "Cojera posterior → asimetría del sacro")]):
    m = med[med.experimento == exp].pivot(index="id_caballo", columns="cojo", values=var)
    for h, r in m.iterrows():
        color = "#C44E52" if r[1] > r[0] else "#7f7f7f"
        ax.plot([0, 1], [r[0], r[1]], "o-", color=color, alpha=.8)
        ax.text(1.03, r[1], h.split("_")[1], fontsize=8, va="center")
    ax.set_xticks([0, 1], ["Basal (sano)", "Inducción (cojo)"]); ax.set_xlim(-.2, 1.2)
    ax.set_title(tit); ax.set_ylabel(f"Mediana de {var}")
plt.tight_layout(); plt.savefig(FIG / "imu_pareado_caballos.png"); plt.show()

# %% [markdown]
# **Lectura ("hacer hablar a los números"):** el patrón coincide con la biomecánica descrita en la literatura veterinaria:
# - En **cojera anterior**, el caballo "levanta la cabeza" al apoyar el miembro doloroso → se duplica la asimetría
#   vertical de la **cabeza** (MinDiff), mientras que la pelvis casi no cambia.
# - En **cojera posterior**, la asimetría aparece en el **sacro** y en la diferencia de movimiento entre tuberosidades
#   coxales (*hip hike*). La cabeza también cambia un poco (movimiento compensatorio).
# - La respuesta **no es igual en todos los caballos**: algunos casi no cambian. Eso anticipa la dificultad de
#   generalizar a caballos nuevos y justifica perfilar a los individuos (sección 6).

# %%
plt.figure(figsize=(10, 8))
corr = df[FEATS].corr(method="spearman")
sns.heatmap(corr, cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink": .7}, xticklabels=True, yticklabels=True)
plt.xticks(fontsize=7); plt.yticks(fontsize=7)
plt.title("Correlación de Spearman entre variables de marcha")
plt.savefig(FIG / "imu_correlaciones.png"); plt.show()
alta = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
print("Pares con |rho| > 0.85 (redundancia):\n", alta[alta.abs() > 0.85].round(2))

# %% [markdown]
# ## 4. Outliers estadísticos: univariados (IQR) y multivariados (Isolation Forest)
# Los artefactos físicos ya se quitaron. Los outliers que quedan son **plausibles** y se analizan, no se borran:
# ¿son zancadas especialmente cojas, caballos atípicos o ruido?

# %%
Q1, Q3 = df[clave].quantile(.25), df[clave].quantile(.75)
out_iqr = ((df[clave] < Q1 - 1.5 * (Q3 - Q1)) | (df[clave] > Q3 + 1.5 * (Q3 - Q1)))
df["n_flags_iqr"] = out_iqr.sum(axis=1)
Xs = StandardScaler().fit_transform(df[FEATS])
iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=SEED).fit(Xs)
df["outlier_iforest"] = iso.predict(Xs) == -1
df["score_anomalia"] = -iso.score_samples(Xs)
print("Outliers IQR (≥1 variable):", (df.n_flags_iqr > 0).sum(), "| Isolation Forest:", df.outlier_iforest.sum())
pd.crosstab([df.experimento, df.estado], df.outlier_iforest, normalize="index").mul(100).round(1)

# %%
top_h = df[df.outlier_iforest].groupby("id_caballo").size().sort_values(ascending=False)
print("Caballos con más zancadas anómalas:\n", top_h.head(6))
df[df.outlier_iforest].groupby("estado")[clave].median().T.round(1)

# %% [markdown]
# **Interpretación:**
# - El criterio IQR marca ~1 de cada 4 zancadas en al menos una variable: con 8 variables asimétricas (cola larga)
#   es esperable, y por eso se prefiere un criterio **multivariado** (Isolation Forest, 5 %).
# - Las anomalías multivariadas son casi el doble de frecuentes en la **cojera posterior inducida** (7,3 % frente a 4,1 %)
#   y tienen asimetrías mayores y harmonic ratio más bajo: buena parte del "outlier" es **la señal clínica**, no ruido.
# - Un solo caballo (`pos_B`) acumula ~25 % de las anomalías: es un **perfil individual atípico**, con movimiento muy
#   irregular incluso en la línea base. Se conserva, pero se vigila su efecto en la validación por caballo.

# %% [markdown]
# ## 5. PCA: estructura del espacio de características

# %%
scaler = StandardScaler().fit(df[FEATS])
pca = PCA(random_state=SEED).fit(scaler.transform(df[FEATS]))
Z = pca.transform(scaler.transform(df[FEATS]))
ev = pca.explained_variance_ratio_
n90 = int(np.argmax(ev.cumsum() >= .90) + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
axes[0].bar(range(1, len(ev) + 1), ev * 100, alpha=.7, label="Individual")
axes[0].plot(range(1, len(ev) + 1), ev.cumsum() * 100, "o-", color="#C44E52", label="Acumulada")
axes[0].axhline(90, ls="--", c="gray"); axes[0].set(xlabel="Componente", ylabel="% varianza", title=f"Scree plot (90 % con {n90} componentes)")
axes[0].legend()
for (e, est), g in df.groupby(["experimento", "estado"]):
    idx = g.index
    axes[1].scatter(Z[idx, 0], Z[idx, 1], s=7, alpha=.5, label=f"{e} · {est}",
                    marker="o" if e == "anterior" else "^")
load = pd.DataFrame(pca.components_[:2].T, index=FEATS, columns=["PC1", "PC2"])
for f in load.abs().sum(axis=1).sort_values(ascending=False).index[:6]:
    axes[1].arrow(0, 0, load.loc[f, "PC1"] * 6, load.loc[f, "PC2"] * 6, color="black", head_width=.12)
    axes[1].text(load.loc[f, "PC1"] * 6.6, load.loc[f, "PC2"] * 6.6, f, fontsize=7)
axes[1].set(xlabel=f"PC1 ({ev[0]*100:.1f} %)", ylabel=f"PC2 ({ev[1]*100:.1f} %)", title="Biplot PCA")
axes[1].legend(fontsize=7, markerscale=2)
plt.tight_layout(); plt.savefig(FIG / "imu_pca.png"); plt.show()
load.assign(abs_sum=load.abs().sum(axis=1)).sort_values("abs_sum", ascending=False).head(8).round(2)

# %% [markdown]
# ## 6. Clustering: perfiles de marcha
# ### 6.1 K-Means sobre las componentes principales (nivel zancada)

# %%
Zk = Z[:, :n90]
sil = {k: silhouette_score(Zk, KMeans(k, n_init=10, random_state=SEED).fit_predict(Zk), sample_size=1500, random_state=SEED)
       for k in range(2, 9)}
k_best = max(sil, key=sil.get)
print("Silhouette por k:", {k: round(v, 3) for k, v in sil.items()}, "→ k =", k_best)
df["perfil_marcha"] = KMeans(k_best, n_init=20, random_state=SEED).fit_predict(Zk)
display(pd.crosstab([df.experimento, df.estado], df.perfil_marcha, normalize="index").mul(100).round(1))
df.groupby("perfil_marcha")[clave].median().round(1)

# %% [markdown]
# ### 6.2 Clustering jerárquico de caballos (perfil individual por condición)
# Cada punto es un **caballo en una condición** (mediana de sus zancadas). Este nivel es el que interesa en clínica:
# ¿qué caballos se parecen entre sí y cuáles se separan con la cojera?

# %%
perf = df.groupby(["id_caballo", "estado"])[clave].median()
Zh = linkage(StandardScaler().fit_transform(perf), method="ward")
plt.figure(figsize=(14, 4.5))
dendrogram(Zh, labels=[f"{i}|{'C' if 'Cojo' in e else 'S'}" for i, e in perf.index], leaf_font_size=8, color_threshold=Zh[-3, 2])
plt.title("Dendrograma de perfiles caballo-condición (Ward) · S = sano, C = cojo"); plt.ylabel("Distancia")
plt.savefig(FIG / "imu_dendrograma_caballos.png"); plt.show()
perf["grupo_jerarquico"] = fcluster(Zh, 3, criterion="maxclust")
pd.crosstab(perf.index.get_level_values("estado"), perf["grupo_jerarquico"])

# %% [markdown]
# **Lectura:** el clustering, que no conoce las etiquetas, separa parcialmente lo sano de lo cojo. Los grupos que mezclan
# ambas condiciones corresponden a caballos con una asimetría basal alta (sanos "asimétricos" por naturaleza) o con una
# respuesta débil a la inducción. Para el sistema esto implica que **un umbral fijo no basta**: conviene comparar a cada
# caballo con su propia línea base, y el modelo supervisado debe validarse con caballos que no haya visto.

# %%
df.to_csv(OUT / "imu_zancadas_limpio.csv", index=False)
print("Guardado imu_zancadas_limpio.csv", df.shape)
