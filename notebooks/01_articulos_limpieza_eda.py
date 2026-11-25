# %% [markdown]
# # 01 · Corpus científico: limpieza y EDA
# Entrada: `data/raw/openalex_works_raw.csv` (5 consultas a la API de OpenAlex). Salida: `data/processed/articulos_limpio.csv`.

# %%
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW = ROOT / "data" / "raw" / "openalex_works_raw.csv"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figuras" / "01_articulos_eda"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})
ANIO_CORTE = 2026  # año de la extracción

raw = pd.read_csv(RAW)
print("Dimensión cruda:", raw.shape)
raw.head(3)

# %% [markdown]
# ## 1. Diagnóstico de calidad de los datos crudos

# %%
diag = pd.DataFrame({
    "nulos_%": (raw.isna().mean() * 100).round(1),
    "tipo": raw.dtypes.astype(str),
    "valores_unicos": raw.nunique(),
}).sort_values("nulos_%", ascending=False)
print("Registros duplicados por ID (mismo artículo en varias consultas):",
      raw.duplicated("openalex_id").sum())
diag

# %% [markdown]
# ## 2. Limpieza y transformación
# Duplicados, tipos no científicos, retractados, ventana 1990–2026 y duplicados por título.

# %%
df = raw.copy()
consultas = df.groupby("openalex_id")["consulta"].apply(lambda s: ";".join(sorted(set(s))))
df = df.drop_duplicates("openalex_id").drop(columns="consulta").merge(consultas, on="openalex_id")
df["n_consultas"] = df["consulta"].str.count(";") + 1
log = [("Registros únicos por ID", len(df))]

excluir_tipo = ["peer-review", "erratum", "paratext", "letter", "editorial", "other", "supplementary-materials"]
df = df[~df["tipo"].isin(excluir_tipo)]; log.append(("Tras excluir tipos no científicos", len(df)))
df = df[df["retractado"] != True]; log.append(("Tras excluir retractados", len(df)))
df = df[df["anio"].between(1990, ANIO_CORTE)]; log.append(("Ventana 1990–2026", len(df)))

df["titulo_norm"] = df["titulo"].fillna("").str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
df = df[df["titulo_norm"].str.len() > 10]
df = df.sort_values("citas", ascending=False).drop_duplicates("titulo_norm")
log.append(("Tras quitar duplicados por título", len(df)))

# 7. Filtro temático: al menos un término del dominio
_txt = (df["titulo"].fillna("") + " " + df["resumen"].fillna("") + " " + df["keywords"].fillna("")).str.lower()
DOMINIO = r"horse|equine|equid|pony|lame|gait|locomot|pose estimation|keypoint|posture|kinemat|limb|stride|animal behavio"
df = df[_txt.str.contains(DOMINIO, regex=True)]
# Quita estudios de cinemática espermática
df = df[~_txt.loc[df.index].str.contains(r"sperm|semen|spermat", regex=True)]
log.append(("Tras filtro de pertinencia temática", len(df)))

df["paises"] = df["paises"].fillna("Desconocido")
df["pais_principal"] = df["paises"].str.split(";").str[0]
df["tiene_resumen"] = df["resumen"].notna()
df["acceso_abierto"] = df["acceso_abierto"].fillna(False).astype(bool)
df["revista"] = df["revista"].fillna("Sin fuente registrada")
df["keywords"] = df["keywords"].fillna("")

pd.DataFrame(log, columns=["Paso", "Registros"])

# %% [markdown]
# ## 3. Atributos derivados (ingeniería de variables)

# %%
df["edad"] = (ANIO_CORTE - df["anio"]).clip(lower=0) + 1
df["citas_por_anio"] = df["citas"] / df["edad"]
df["log_citas"] = np.log1p(df["citas"])
df["n_keywords"] = df["keywords"].apply(lambda s: len([k for k in s.split(";") if k]))
df["long_resumen"] = df["resumen"].fillna("").str.split().str.len()
df["texto"] = (df["titulo"].fillna("") + ". " + df["resumen"].fillna("") + " " + df["keywords"].str.replace(";", " "))

TEMAS = {
    "vision": r"computer vision|deep learning|convolutional|pose estimation|keypoint|markerless|video|image",
    "sensores": r"inertial|accelerom|gyroscop|\bimu\b|sensor",
    "ml": r"machine learning|random forest|support vector|neural network|classifier|classification",
    "clinico": r"diagnos|nerve block|analgesi|radiograph|ultrason|treatment|clinical",
}
for t, pat in TEMAS.items():
    df[f"tema_{t}"] = df["texto"].str.lower().str.contains(pat, regex=True)
df["periodo"] = pd.cut(df["anio"], [1989, 2005, 2015, 2020, ANIO_CORTE],
                       labels=["1990–2005", "2006–2015", "2016–2020", "2021–2026"])
print("Dataset limpio:", df.shape)
df[["titulo", "anio", "citas", "citas_por_anio", "n_consultas", "tema_vision", "tema_sensores"]].head()

# %% [markdown]
# ## 4. Estadística descriptiva

# %%
num = ["anio", "citas", "citas_por_anio", "fwci", "n_autores", "n_paises", "n_referencias", "n_keywords", "long_resumen"]
desc = df[num].describe().T
desc["asimetria"] = df[num].skew()
desc["curtosis"] = df[num].kurt()
desc.round(2)

# %% [markdown]
# ## 5. Visualizaciones

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
anual = df.groupby("anio").size()
axes[0].bar(anual.index, anual.values, color="#4C72B0")
axes[0].set(title="Publicaciones por año (corpus limpio)", xlabel="Año", ylabel="N.º de trabajos")
temas_anio = df[df["anio"] >= 2000].groupby("anio")[[f"tema_{t}" for t in TEMAS]].mean() * 100
temas_anio.columns = ["Visión por computador", "Sensores inerciales", "Machine learning", "Clínico"]
temas_anio.rolling(3, center=True).mean().plot(ax=axes[1], lw=2)
axes[1].set(title="Proporción de trabajos por enfoque (media móvil 3 años)", xlabel="Año", ylabel="% de trabajos del año")
plt.tight_layout(); plt.savefig(FIG / "art_tendencia_anual.png"); plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["citas"], bins=60, ax=axes[0], color="#C44E52")
axes[0].set(title="Citas (escala original)", xlabel="Citas", ylabel="Frecuencia")
sns.histplot(df["log_citas"], bins=40, kde=True, ax=axes[1], color="#55A868")
axes[1].set(title="log(1 + citas)", xlabel="log(1 + citas)")
stats.probplot(df["log_citas"], dist="norm", plot=axes[2]); axes[2].set_title("Q-Q plot de log(1 + citas)")
plt.tight_layout(); plt.savefig(FIG / "art_distribucion_citas.png"); plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
paises = df.loc[df["pais_principal"] != "Desconocido", "pais_principal"].value_counts().head(15)
sns.barplot(x=paises.values, y=paises.index, ax=axes[0], color="#8172B2")
axes[0].set(title="Top 15 países (primer autor con afiliación)", xlabel="N.º de trabajos", ylabel="")
rev = df.loc[df["revista"] != "Sin fuente registrada", "revista"].value_counts().head(12)
sns.barplot(x=rev.values, y=[r[:45] for r in rev.index], ax=axes[1], color="#64B5CD")
axes[1].set(title="Top 12 fuentes de publicación", xlabel="N.º de trabajos", ylabel="")
plt.tight_layout(); plt.savefig(FIG / "art_paises_revistas.png"); plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
orden = df["tipo"].value_counts().index
sns.boxplot(data=df, x="log_citas", y="tipo", order=orden, ax=axes[0], color="#DD8452", fliersize=2)
axes[0].set(title="Impacto por tipo de documento", xlabel="log(1 + citas)", ylabel="")
oa = df.groupby("periodo")["acceso_abierto"].mean() * 100
axes[1].bar(oa.index.astype(str), oa.values, color="#4C72B0")
for i, v in enumerate(oa.values):
    axes[1].text(i, v + 1, f"{v:.0f}%", ha="center")
axes[1].set(title="Porcentaje en acceso abierto por periodo", ylabel="%", ylim=(0, 100))
plt.tight_layout(); plt.savefig(FIG / "art_tipo_oa.png"); plt.show()

# %%
corr = df[["anio", "log_citas", "citas_por_anio", "fwci", "n_autores", "n_paises", "n_referencias", "n_keywords", "long_resumen", "n_consultas"]].corr(method="spearman")
plt.figure(figsize=(8.5, 6.8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink": .8})
plt.title("Correlación de Spearman entre variables numéricas")
plt.savefig(FIG / "art_correlaciones.png"); plt.show()

# %% [markdown]
# ## 6. Outliers
# IQR sobre `log(1+citas_por_anio)` y z-score robusto (mediana/MAD). Se marcan, no se eliminan.

# %%
x = np.log1p(df["citas_por_anio"])
q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
df["outlier_iqr"] = x > q3 + 1.5 * iqr
mad = np.median(np.abs(x - x.median()))
df["z_robusto"] = 0.6745 * (x - x.median()) / mad
df["outlier_z"] = df["z_robusto"].abs() > 3.5
df["outlier"] = df["outlier_iqr"] | df["outlier_z"]
print(f"Outliers IQR: {df.outlier_iqr.sum()} | z robusto: {df.outlier_z.sum()} | unión: {df.outlier.sum()}")

top_out = df[df["outlier"]].sort_values("citas_por_anio", ascending=False)
top_out[["titulo", "anio", "citas", "citas_por_anio", "consulta", "tipo"]].head(15)

# %%
comp = df.groupby("outlier")[[f"tema_{t}" for t in TEMAS] + ["acceso_abierto", "n_autores", "n_paises", "n_consultas"]].mean().T
comp.columns = ["Normal", "Outlier"]
comp["razon_outlier/normal"] = comp["Outlier"] / comp["Normal"]
comp.round(2)

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.scatter(df.loc[~df.outlier, "anio"], df.loc[~df.outlier, "citas_por_anio"], s=6, alpha=.35, label="Normal")
ax.scatter(df.loc[df.outlier, "anio"], df.loc[df.outlier, "citas_por_anio"], s=14, c="#C44E52", label="Outlier (IQR ∪ z robusto)")
ax.set_yscale("symlog", linthresh=1)
ax.set(title="Impacto ajustado por antigüedad: outliers", xlabel="Año", ylabel="Citas por año (escala symlog)")
ax.legend()
plt.savefig(FIG / "art_outliers.png"); plt.show()

# %% [markdown]
# ## 7. Exportación del dataset limpio

# %%
cols = ["openalex_id", "doi", "titulo", "anio", "periodo", "tipo", "idioma", "citas", "log_citas", "citas_por_anio", "fwci",
        "acceso_abierto", "revista", "n_autores", "paises", "pais_principal", "n_paises", "n_referencias", "keywords",
        "n_keywords", "conceptos", "topico", "subcampo", "campo", "tiene_resumen", "long_resumen", "texto", "consulta",
        "n_consultas"] + [f"tema_{t}" for t in TEMAS] + ["z_robusto", "outlier_iqr", "outlier_z", "outlier"]
df[cols].to_csv(OUT / "articulos_limpio.csv", index=False)
print("Guardado:", OUT / "articulos_limpio.csv", df[cols].shape)
