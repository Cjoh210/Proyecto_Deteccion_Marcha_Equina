"""
English versions of the report figures (for the English IEEE paper).
Re-plots each figure from the processed datasets produced by notebooks 01-06, with English labels.

Usage:  python src/figuras_en.py      ->  report/final/en/figures/*.png
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import RocCurveDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
OUT = ROOT / "report" / "final" / "en" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})

PROFILES = {
    "Cinemática y biomecánica de la marcha": "Gait kinematics and biomechanics",
    "Diagnóstico clínico de la cojera": "Clinical lameness diagnosis",
    "Genética, bienestar y rendimiento": "Genetics, welfare and performance",
    "Patologías articulares y tratamiento": "Joint pathology and treatment",
    "Estimación de pose animal (deep learning)": "Animal pose estimation (deep learning)",
    "Sensores inerciales y asimetría de marcha": "Inertial sensors and gait asymmetry",
    "Visión por computador para monitoreo animal": "Computer vision for animal monitoring",
    "Detección de cojera en bovinos": "Lameness detection in cattle",
}
NODES = {"cabeza": "head", "cruz": "withers", "sacro": "sacrum", "tc_izq": "left_hip", "tc_der": "right_hip"}


def feat_en(name):
    if name == "duracion_zancada_s":
        return "stride_duration_s"
    for es, en in NODES.items():
        if name.startswith(es + "_"):
            return en + name[len(es):]
    return name


def save(name):
    plt.savefig(OUT / name)
    plt.close("all")
    print("ok", name)


# ---------------------------------------------------------------- corpus: trends
art = pd.read_csv(P / "articulos_limpio.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
yearly = art.groupby("anio").size()
axes[0].bar(yearly.index, yearly.values, color="#4C72B0")
axes[0].set(title="Publications per year (clean corpus)", xlabel="Year", ylabel="Number of papers")
temas = art[art["anio"] >= 2000].groupby("anio")[["tema_vision", "tema_sensores", "tema_ml", "tema_clinico"]].mean() * 100
temas.columns = ["Computer vision", "Inertial sensors", "Machine learning", "Clinical"]
temas.rolling(3, center=True).mean().plot(ax=axes[1], lw=2)
axes[1].set(title="Share of papers by approach (3-year moving average)", xlabel="Year", ylabel="% of papers in the year")
plt.tight_layout(); save("art_tendencia_anual.png")

# ---------------------------------------------------------------- corpus: t-SNE map
cl = pd.read_csv(P / "articulos_con_clusters.csv")
cl["profile"] = cl["perfil"].map(PROFILES)
plt.figure(figsize=(10.5, 7))
sns.scatterplot(data=cl, x="tsne_x", y="tsne_y", hue="profile", s=9, alpha=.7, linewidth=0)
o = cl[cl["outlier"]]
plt.scatter(o["tsne_x"], o["tsne_y"], s=40, facecolors="none", edgecolors="black", linewidths=.8, label="Impact outlier")
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, markerscale=2)
plt.title("Map of research lines (t-SNE) · K-Means k=8"); plt.xlabel(""); plt.ylabel("")
save("art_tsne_clusters.png")

# ---------------------------------------------------------------- corpus: choosing k
res = pd.read_csv(P / "kmeans_seleccion_k.csv")
fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
for ax, col, t in zip(axes, ["inercia", "silhouette", "calinski", "davies_bouldin"],
                      ["Elbow (inertia ↓)", "Silhouette ↑", "Calinski-Harabasz ↑", "Davies-Bouldin ↓"]):
    ax.plot(res["k"], res[col], "o-"); ax.set_title(t); ax.set_xlabel("k")
plt.tight_layout(); save("art_kmeans_seleccion_k.png")

# ---------------------------------------------------------------- catalogue: suitability matrix
apt = pd.read_csv(P / "catalogo_aptitud.csv", index_col=0)
apt.index = apt.index.map(lambda n: {
    "IMU caballo islandés - cojera inducida": "Icelandic horse IMU - induced lameness",
    "Horse Treadmill (Zúrich / hSMAL)": "Horse Treadmill (Zurich / hSMAL)",
    "Cojera en ovejas (sensor de oreja)": "Sheep lameness (ear sensor)",
    "Collar IMU bovinos (Chile)": "Cattle IMU collar (Chile)"}.get(n, n))
apt.columns = ["Horse species", "Lameness label", "Video / temporal", "Anatomical keypoints",
               "Scale (log samples)", "License / access", "Score"]
plt.figure(figsize=(9, 6.2))
sns.heatmap(apt, annot=True, fmt=".2f", cmap="YlGn", cbar=False, linewidths=.5)
plt.title("Suitability of each dataset for vision-based equine lameness detection"); plt.ylabel("")
save("cat_matriz_aptitud.png")

# ---------------------------------------------------------------- IMU
imu = pd.read_csv(P / "imu_zancadas_limpio.csv")
imu["state"] = imu["cojo"].map({0: "Sound (BAS)", 1: "Lame (IND)"})
imu["experiment"] = imu["experimento"].map({"anterior": "forelimb", "posterior": "hindlimb"})
FEATS = [c for c in imu.columns if c.endswith(("_mm", "_hr", "_rms")) or c == "duracion_zancada_s"]
key = ["cabeza_mindiff_mm", "cabeza_maxdiff_mm", "sacro_mindiff_mm", "sacro_maxdiff_mm", "hip_hike_diff_mm",
       "cabeza_hr", "sacro_hr", "duracion_zancada_s"]

fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
for ax, c in zip(axes.ravel(), key):
    sns.boxplot(data=imu, x="experiment", y=c, hue="state", ax=ax, fliersize=1.5,
                palette={"Sound (BAS)": "#4C72B0", "Lame (IND)": "#C44E52"})
    ax.set_title(feat_en(c)); ax.set_xlabel(""); ax.set_ylabel("")
    if ax is not axes[0, 0]:
        ax.get_legend().remove()
plt.suptitle("Per-stride symmetry metrics: sound vs. induced lameness", y=1.01)
plt.tight_layout(); save("imu_boxplots.png")

med = imu.groupby(["experimento", "id_caballo", "cojo"])[key].median().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, (exp, var, title) in zip(axes, [("anterior", "cabeza_mindiff_mm", "Forelimb lameness → head asymmetry"),
                                        ("posterior", "sacro_mindiff_mm", "Hindlimb lameness → pelvic asymmetry")]):
    m = med[med.experimento == exp].pivot(index="id_caballo", columns="cojo", values=var)
    for h, r in m.iterrows():
        ax.plot([0, 1], [r[0], r[1]], "o-", color="#C44E52" if r[1] > r[0] else "#7f7f7f", alpha=.8)
        ax.text(1.03, r[1], h.split("_")[1], fontsize=8, va="center")
    ax.set_xticks([0, 1], ["Baseline (sound)", "Induction (lame)"]); ax.set_xlim(-.2, 1.2)
    ax.set_title(title); ax.set_ylabel(f"Median {feat_en(var)}")
plt.tight_layout(); save("imu_pareado_caballos.png")

sc = StandardScaler().fit(imu[FEATS]); Z = sc.transform(imu[FEATS])
pca = PCA(random_state=42).fit(Z); S = pca.transform(Z); ev = pca.explained_variance_ratio_
n90 = int(np.argmax(ev.cumsum() >= .90) + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
axes[0].bar(range(1, len(ev) + 1), ev * 100, alpha=.7, label="Individual")
axes[0].plot(range(1, len(ev) + 1), ev.cumsum() * 100, "o-", color="#C44E52", label="Cumulative")
axes[0].axhline(90, ls="--", c="gray"); axes[0].set(xlabel="Component", ylabel="% variance", title=f"Scree plot (90 % with {n90} components)")
axes[0].legend()
for (e, st), g in imu.groupby(["experiment", "state"]):
    axes[1].scatter(S[g.index, 0], S[g.index, 1], s=7, alpha=.5, label=f"{e} · {st}", marker="o" if e == "forelimb" else "^")
load = pd.DataFrame(pca.components_[:2].T, index=FEATS, columns=["PC1", "PC2"])
for f in load.abs().sum(axis=1).sort_values(ascending=False).index[:6]:
    axes[1].arrow(0, 0, load.loc[f, "PC1"] * 6, load.loc[f, "PC2"] * 6, color="black", head_width=.12)
    axes[1].text(load.loc[f, "PC1"] * 6.6, load.loc[f, "PC2"] * 6.6, feat_en(f), fontsize=7)
axes[1].set(xlabel=f"PC1 ({ev[0]*100:.1f} %)", ylabel=f"PC2 ({ev[1]*100:.1f} %)", title="PCA biplot")
axes[1].legend(fontsize=7, markerscale=2)
plt.tight_layout(); save("imu_pca.png")

# ---------------------------------------------------------------- ML classifier
pr = pd.read_csv(P / "ml_probabilidades_test.csv")
names = {"Random Forest": "Random Forest", "Gradient Boosting": "Gradient Boosting", "SVM (RBF)": "SVM (RBF)",
         "Regresión logística": "Logistic regression"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for col, en in names.items():
    if col in pr:
        RocCurveDisplay.from_predictions(pr["y_true"], pr[col], name=en, ax=axes[0])
axes[0].plot([0, 1], [0, 1], "--", c="gray"); axes[0].set_title("ROC curves · test horses")
cm = confusion_matrix(pr["y_true"], (pr["Random Forest"] >= .5).astype(int))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1], xticklabels=["Sound", "Lame"], yticklabels=["Sound", "Lame"])
axes[1].set(title="Confusion matrix · Random Forest", xlabel="Predicted", ylabel="Actual")
plt.tight_layout(); save("ml_roc_confusion.png")

meta = json.load(open(ROOT / "models" / "modelo_cojera_meta.json"))
model = joblib.load(ROOT / "models" / "modelo_cojera.joblib")
te = imu[imu["id_caballo"].isin(meta["caballos_test"])]
Xt = pd.DataFrame(model[:-1].transform(te[meta["features"]]), columns=[feat_en(f) for f in meta["features"]])
sv = shap.TreeExplainer(model.named_steps["clf"]).shap_values(Xt)
sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
plt.figure()
shap.summary_plot(sv, Xt, max_display=12, show=False)
plt.title("SHAP · Random Forest (test horses)")
save("ml_shap_summary.png")

# ---------------------------------------------------------------- regression
rr = pd.read_csv(P / "regresion_resultados.csv")
rr["model"] = rr["modelo"].replace({"Lineal": "Linear", "Polinomial (grado 2) + Ridge": "Polynomial (degree 2) + Ridge",
                                    "Random Forest (referencia)": "Random Forest (reference)"})
pp = pd.read_csv(P / "regresion_predicciones_test.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
r = rr.sort_values("R2_test")
axes[0].barh(r["model"], r["R2_test"], color=["#C44E52" if "Random" in m else "#4C72B0" for m in r["model"]])
axes[0].set(title="R² on the test set", xlabel="R²")
axes[1].scatter(pp["y_real"], pp["y_pred"], s=8, alpha=.4)
lim = [0, max(pp["y_real"].max(), pp["y_pred"].max())]
axes[1].plot(lim, lim, "--", c="gray")
axes[1].set(title="Actual vs. predicted · best linear-family model", xlabel="log(1 + citations/year) actual", ylabel="Predicted")
plt.tight_layout(); save("reg_comparacion.png")

coef = pd.read_csv(P / "regresion_lasso_coeficientes.csv", index_col=0)["coeficiente"]
VARS = {"anio": "year", "n_autores": "number of authors", "n_paises": "number of countries", "n_referencias": "number of references",
        "n_keywords": "number of keywords", "long_resumen": "abstract length", "n_consultas": "number of matching queries",
        "acceso_abierto": "open access", "tiene_resumen": "has abstract", "tema_vision": "topic: computer vision",
        "tema_sensores": "topic: sensors", "tema_ml": "topic: machine learning", "tema_clinico": "topic: clinical"}
def var_en(v):
    if v.startswith("perfil_"):
        return "profile: " + PROFILES.get(v[7:], v[7:])
    if v.startswith("tipo_"):
        t = v[5:]
        return "type: " + ("other" if "infrequent" in t else t)
    return VARS.get(v, v)
coef.index = coef.index.map(var_en)
coef = coef.sort_values()
plt.figure(figsize=(8, 0.32 * len(coef) + 1))
plt.barh(coef.index, coef.values, color=np.where(coef.values > 0, "#55A868", "#C44E52"))
plt.axvline(0, c="black", lw=.8)
plt.title("Lasso coefficients (standardized variables)"); plt.xlabel("Effect on log(1 + citations/year)")
save("reg_lasso_coeficientes.png")
