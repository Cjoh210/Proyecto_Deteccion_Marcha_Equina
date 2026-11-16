# %% [markdown]
# # 04 · Machine Learning (avance preliminar): clasificación sano vs. cojo por zancada
# Fase CRISP-DM: *Modelado* y *Evaluación*. Entrada: `data/processed/imu_zancadas_limpio.csv` (notebook 03).
#
# **Pregunta predictiva:** dadas las métricas de simetría de una zancada de trote, ¿está el caballo cojo?
#
# **Decisión metodológica clave: validación agrupada por caballo.** Las zancadas de un mismo caballo se parecen mucho
# entre sí. Si se mezclaran en entrenamiento y prueba, el modelo "reconocería al caballo" y la métrica saldría inflada.
# En un caso real el sistema evaluará **caballos que nunca vio**, así que:
# - *Hold-out*: 5 caballos completos (≈ 23 %) se reservan como **prueba final**.
# - Selección de modelo e hiperparámetros: **GroupKFold (5)** dentro de los 17 caballos de entrenamiento.
# - Robustez: **Leave-One-Horse-Out** sobre los 22 caballos para el modelo final.

# %%
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit, LeaveOneGroupOut,
                                     RandomizedSearchCV, cross_validate)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import loguniform

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG = ROOT / "docs" / "figures"; OUT = ROOT / "data" / "processed"; MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})
SEED = 42

df = pd.read_csv(OUT / "imu_zancadas_limpio.csv")
FEATS = [c for c in df.columns if c.endswith(("_mm", "_hr", "_rms")) or c == "duracion_zancada_s"]
X, y, g = df[FEATS], df["cojo"], df["id_caballo"]
print(f"Zancadas: {len(df)} | variables: {len(FEATS)} | caballos: {g.nunique()} | % cojo: {y.mean()*100:.1f}")

# %% [markdown]
# ## 1. Partición train/test por caballo (estratificada por experimento)

# %%
gss = GroupShuffleSplit(n_splits=200, test_size=5, random_state=SEED)
for tr, te in gss.split(X, y, g):
    exp_te = df.iloc[te].groupby("experimento")["id_caballo"].nunique()
    if len(exp_te) == 2 and exp_te.min() >= 2 and 0.4 < y.iloc[te].mean() < 0.6:
        break  # prueba con caballos de ambos experimentos y clases balanceadas
X_tr, X_te, y_tr, y_te, g_tr, g_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], g.iloc[tr], g.iloc[te]
print("Caballos de prueba:", sorted(g_te.unique()))
print(f"Train: {len(X_tr)} zancadas ({g_tr.nunique()} caballos) | Test: {len(X_te)} zancadas ({g_te.nunique()} caballos)")
cv = GroupKFold(n_splits=5)

# %% [markdown]
# ## 2. Modelos base y ensembles con el mismo pipeline
# `SimpleImputer(mediana) → StandardScaler → modelo`. El escalado se ajusta solo con los datos de entrenamiento de
# cada fold, así que no hay fuga de información. Métrica principal: **ROC-AUC** (no depende del umbral y es estable
# con clases casi balanceadas). También se reportan F1 y exactitud balanceada.

# %%
def pipe(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", model)])

modelos = {
    "Regresión logística": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "KNN (k=15)": KNeighborsClassifier(n_neighbors=15),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
scoring = {"roc_auc": "roc_auc", "f1": "f1", "bal_acc": "balanced_accuracy"}
filas = []
for nombre, m in modelos.items():
    t0 = time.time()
    r = cross_validate(pipe(m), X_tr, y_tr, groups=g_tr, cv=cv, scoring=scoring, n_jobs=-1)
    filas.append({"modelo": nombre, **{f"{k}_media": r[f"test_{k}"].mean() for k in scoring},
                  "roc_auc_std": r["test_roc_auc"].std(), "tiempo_s": time.time() - t0})
base = pd.DataFrame(filas).sort_values("roc_auc_media", ascending=False)
base.round(3)

# %% [markdown]
# ## 3. Optimización de hiperparámetros
# - **Grid Search** para Random Forest y Gradient Boosting (espacios pequeños y discretos).
# - **Random Search** para SVM (C y gamma son continuos en escala logarítmica, donde una búsqueda aleatoria es más
#   eficiente que una grilla).
# Siempre con GroupKFold por caballo.

# %%
busquedas = {
    "Random Forest": GridSearchCV(pipe(RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)), {
        "clf__n_estimators": [200, 500], "clf__max_depth": [4, 8, None],
        "clf__min_samples_leaf": [1, 5, 20], "clf__max_features": ["sqrt", 0.5]},
        scoring="roc_auc", cv=cv, n_jobs=-1),
    "Gradient Boosting": GridSearchCV(pipe(GradientBoostingClassifier(random_state=SEED)), {
        "clf__n_estimators": [100, 300], "clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3],
        "clf__subsample": [0.7, 1.0]},
        scoring="roc_auc", cv=cv, n_jobs=-1),
    "SVM (RBF)": RandomizedSearchCV(pipe(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)), {
        "clf__C": loguniform(1e-2, 1e2), "clf__gamma": loguniform(1e-4, 1e0)},
        n_iter=30, scoring="roc_auc", cv=cv, n_jobs=-1, random_state=SEED),
}
opt = []
for nombre, s in busquedas.items():
    t0 = time.time(); s.fit(X_tr, y_tr, groups=g_tr)
    opt.append({"modelo": nombre, "roc_auc_cv": s.best_score_, "mejores_parametros": s.best_params_, "tiempo_s": time.time() - t0})
opt = pd.DataFrame(opt).sort_values("roc_auc_cv", ascending=False)
pd.set_option("display.max_colwidth", 200)
opt.round(3)

# %% [markdown]
# ## 4. Evaluación final en los 5 caballos de prueba (nunca vistos)

# %%
res_test, probas = [], {}
for nombre, s in busquedas.items():
    p = s.best_estimator_.predict_proba(X_te)[:, 1]; probas[nombre] = p
    pred = (p >= .5).astype(int)
    res_test.append({"modelo": nombre, "roc_auc": roc_auc_score(y_te, p), "f1": f1_score(y_te, pred),
                     "bal_acc": balanced_accuracy_score(y_te, pred)})
lr = pipe(modelos["Regresión logística"]).fit(X_tr, y_tr); p = lr.predict_proba(X_te)[:, 1]; probas["Regresión logística"] = p
res_test.append({"modelo": "Regresión logística (base)", "roc_auc": roc_auc_score(y_te, p),
                 "f1": f1_score(y_te, p >= .5), "bal_acc": balanced_accuracy_score(y_te, p >= .5)})
res_test = pd.DataFrame(res_test).sort_values("roc_auc", ascending=False)
res_test.round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for nombre, p in probas.items():
    RocCurveDisplay.from_predictions(y_te, p, name=nombre, ax=axes[0])
axes[0].plot([0, 1], [0, 1], "--", c="gray"); axes[0].set_title("Curvas ROC · caballos de prueba")
# Modelo final: el mejor ENSEMBLE según la validación cruzada (requisito R3). La SVM queda muy cerca en CV
# (diferencia menor que la desviación entre folds), pero el ensemble se puede explicar con SHAP de forma exacta (TreeSHAP).
mejor = opt[opt.modelo.isin(["Random Forest", "Gradient Boosting"])].iloc[0]["modelo"]
print("Modelo final seleccionado por CV:", mejor)
cm = confusion_matrix(y_te, (probas[mejor] >= .5).astype(int))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1], xticklabels=["Sano", "Cojo"], yticklabels=["Sano", "Cojo"])
axes[1].set(title=f"Matriz de confusión · {mejor}", xlabel="Predicho", ylabel="Real")
plt.tight_layout(); plt.savefig(FIG / "ml_roc_confusion.png"); plt.show()
print(classification_report(y_te, (probas[mejor] >= .5).astype(int), target_names=["Sano", "Cojo"]))

# %% [markdown]
# ### Decisión a nivel de prueba (trial)
# Clínicamente no se diagnostica una zancada sino una prueba de trote completa. Se promedia la probabilidad de todas
# las zancadas de cada prueba (caballo × condición) y se clasifica la prueba.

# %%
te_df = df.iloc[te][["id_caballo", "experimento", "condicion", "cojo"]].copy()
te_df["p_cojo"] = probas[mejor]
trial = te_df.groupby(["id_caballo", "experimento", "condicion"]).agg(real=("cojo", "first"), p_media=("p_cojo", "mean"),
                                                                        n_zancadas=("p_cojo", "size")).reset_index()
trial["pred"] = (trial["p_media"] >= .5).astype(int)
trial["acierto"] = trial["pred"] == trial["real"]
print(f"Exactitud por prueba: {trial.acierto.mean()*100:.0f} % ({trial.acierto.sum()}/{len(trial)})")
trial.round(3)

# %% [markdown]
# ## 5. Robustez: Leave-One-Horse-Out (22 caballos)

# %%
final = clone(busquedas[mejor].best_estimator_)
logo = LeaveOneGroupOut()
r = cross_validate(final, X, y, groups=g, cv=logo, scoring={"bal_acc": "balanced_accuracy"}, n_jobs=-1, return_estimator=False)
# AUC por caballo no está definida (una sola clase si falta una condición); se calcula AUC global con predicciones fuera de fold
from sklearn.model_selection import cross_val_predict
p_oof = cross_val_predict(final, X, y, groups=g, cv=logo, method="predict_proba", n_jobs=-1)[:, 1]
df["p_oof"] = p_oof
print(f"LOHO · ROC-AUC global (out-of-fold): {roc_auc_score(y, p_oof):.3f} | exactitud balanceada media: {r['test_bal_acc'].mean():.3f}")
por_caballo = df.groupby(["experimento", "id_caballo"]).apply(
    lambda d: pd.Series({"acc_zancada": ((d.p_oof >= .5) == d.cojo).mean(),
                         "p_media_sano": d.loc[d.cojo == 0, "p_oof"].mean(), "p_media_cojo": d.loc[d.cojo == 1, "p_oof"].mean()}))
fig, ax = plt.subplots(figsize=(12, 3.8))
pc = por_caballo.reset_index()
ax.bar(pc["id_caballo"], pc["acc_zancada"], color=np.where(pc.experimento == "anterior", "#4C72B0", "#DD8452"))
ax.axhline(.5, ls="--", c="gray"); ax.set_ylim(0, 1)
ax.set(title="Exactitud por caballo en Leave-One-Horse-Out (azul: anterior · naranja: posterior)", ylabel="Exactitud")
plt.xticks(rotation=60); plt.savefig(FIG / "ml_loho_caballos.png"); plt.show()
por_caballo.round(2)

# %% [markdown]
# ## 6. Interpretabilidad (SHAP + importancia por permutación)

# %%
est = busquedas[mejor].best_estimator_
nombre_int = mejor
Xte_t = pd.DataFrame(est[:-1].transform(X_te), columns=FEATS)
explainer = shap.TreeExplainer(est.named_steps["clf"])
sv = explainer.shap_values(Xte_t)
sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
plt.figure()
shap.summary_plot(sv, Xte_t, max_display=12, show=False)
plt.title(f"SHAP · {nombre_int} (caballos de prueba)")
plt.savefig(FIG / "ml_shap_summary.png"); plt.show()

# %%
pi = permutation_importance(est, X_te, y_te, scoring="roc_auc", n_repeats=20, random_state=SEED, n_jobs=-1)
imp = pd.DataFrame({"variable": FEATS, "perm_importancia": pi.importances_mean, "perm_std": pi.importances_std,
                    "shap_abs_medio": np.abs(sv).mean(axis=0)}).sort_values("shap_abs_medio", ascending=False)
imp.head(12).round(4)

# %% [markdown]
# **Lectura:** las variables que más pesan en el modelo son las mismas que la biomecánica señala como marcadores de
# cojera: la asimetría vertical de la cabeza (MinDiff y harmonic ratio) para la cojera anterior y la del sacro / tuberosidades
# coxales para la posterior. Que el modelo "piense" como un veterinario da confianza para pasar a la fase de visión por
# computador, donde estas mismas señales se estimarán desde video con keypoints de cabeza, cruz y pelvis.

# %%
import joblib, json
joblib.dump(busquedas[mejor].best_estimator_, MODELS / "modelo_cojera.joblib")
json.dump({"modelo": mejor, "features": FEATS, "caballos_test": sorted(g_te.unique().tolist()),
           "roc_auc_test": float(res_test.loc[res_test.modelo == mejor, "roc_auc"].iloc[0]),
           "roc_auc_loho": float(roc_auc_score(y, p_oof))}, open(MODELS / "modelo_cojera_meta.json", "w"), indent=2)
base.to_csv(OUT / "ml_resultados_cv.csv", index=False); res_test.to_csv(OUT / "ml_resultados_test.csv", index=False)
print("Modelo guardado en models/modelo_cojera.joblib")

# %% [markdown]
# ## 7. Conclusiones del avance de ML y siguientes pasos
# - Con variables de **una sola zancada** y validación con **caballos no vistos**, el modelo detecta la cojera inducida
#   mejor que el azar (ROC-AUC ≈ 0,71 tanto en los 5 caballos de prueba como en Leave-One-Horse-Out; 7 de 9 pruebas de trote bien clasificadas al promediar sus zancadas). Es un rendimiento **moderado**, y es
#   honesto: la cojera inducida es leve y algunos caballos son asimétricos incluso sanos.
# - La validación agrupada es la decisión más importante del avance: con una partición aleatoria por zancada las métricas
#   saldrían infladas, porque el modelo reconocería a cada caballo.
# - **Para la entrega final:** (1) normalizar cada caballo frente a una sesión de referencia propia (seguimiento
#   longitudinal), (2) usar asimetrías con signo, identificando el miembro con los sensores de casco, (3) agregar
#   modelos de regresión para estimar el grado de asimetría, y (4) servir el modelo como API en AWS.

