# %% [markdown]
# # 06 · Regresión del impacto de los artículos
# Objetivo: `log(1 + citas_por_anio)`. Modelos: lineal, polinomial, Ridge, Lasso y Elastic Net con Grid Search, y Random Forest como referencia.

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG = ROOT / "figuras" / "06_regresion"; OUT = ROOT / "data" / "processed"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})
SEED = 42

df = pd.read_csv(OUT / "articulos_con_clusters.csv")
df = df[df["anio"] <= 2024].copy()  # 2025-2026 aún sin citas acumuladas
df["y"] = np.log1p(df["citas_por_anio"])
print("Artículos para regresión:", len(df))

NUM = ["anio", "n_autores", "n_paises", "n_referencias", "n_keywords", "long_resumen", "n_consultas"]
BIN = ["acceso_abierto", "tiene_resumen", "tema_vision", "tema_sensores", "tema_ml", "tema_clinico"]
CAT = ["perfil", "tipo"]
for c in BIN:
    df[c] = df[c].astype(int)
df["n_paises"] = df["n_paises"].fillna(0)
X, y = df[NUM + BIN + CAT], df["y"]

# %% [markdown]
# Se excluyen 2025–2026 por tener pocas citas acumuladas.

# %%
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
prep = ColumnTransformer([("num", StandardScaler(), NUM + BIN),
                          ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT)])
cv = KFold(5, shuffle=True, random_state=SEED)

def pipe(model, poly=False):
    pasos = [("prep", prep)]
    if poly:
        pasos.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    return Pipeline(pasos + [("reg", model)])

busquedas = {
    "Lineal": (pipe(LinearRegression()), {}),
    "Polinomial (grado 2) + Ridge": (pipe(Ridge(), poly=True), {"reg__alpha": [10, 100, 1000]}),
    "Ridge": (pipe(Ridge()), {"reg__alpha": np.logspace(-2, 3, 12)}),
    "Lasso": (pipe(Lasso(max_iter=20000)), {"reg__alpha": np.logspace(-4, 0, 12)}),
    "Elastic Net": (pipe(ElasticNet(max_iter=20000)), {"reg__alpha": np.logspace(-4, 0, 8), "reg__l1_ratio": [0.2, 0.5, 0.8]}),
    "Random Forest (referencia)": (pipe(RandomForestRegressor(random_state=SEED, n_jobs=-1)),
                                   {"reg__n_estimators": [300], "reg__max_depth": [6, 12, None], "reg__min_samples_leaf": [5, 20]}),
}
filas, mejores = [], {}
for nombre, (p, grid) in busquedas.items():
    gs = GridSearchCV(p, grid or {"reg__fit_intercept": [True]}, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    mejores[nombre] = gs.best_estimator_
    pred = gs.best_estimator_.predict(X_te)
    filas.append({"modelo": nombre, "RMSE_cv": -gs.best_score_, "R2_test": r2_score(y_te, pred),
                  "RMSE_test": np.sqrt(mean_squared_error(y_te, pred)), "MAE_test": mean_absolute_error(y_te, pred),
                  "parametros": {k.replace("reg__", ""): v for k, v in gs.best_params_.items()}})
res = pd.DataFrame(filas).sort_values("RMSE_cv")
pd.set_option("display.max_colwidth", 120)
res.round(3)

# %% [markdown]
# ## Comparación visual

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
r = res.sort_values("R2_test")
axes[0].barh(r["modelo"], r["R2_test"], color=["#C44E52" if "Random" in m else "#4C72B0" for m in r["modelo"]])
axes[0].set(title="R² en el conjunto de prueba", xlabel="R²")
mejor_lineal = res[~res.modelo.str.contains("Random")].iloc[0]["modelo"]
pred = mejores[mejor_lineal].predict(X_te)
pd.DataFrame({"y_real": y_te.values, "y_pred": pred}).to_csv(OUT / "regresion_predicciones_test.csv", index=False)
axes[1].scatter(y_te, pred, s=8, alpha=.4)
lim = [0, max(y_te.max(), pred.max())]
axes[1].plot(lim, lim, "--", c="gray")
axes[1].set(title=f"Real vs. predicho · {mejor_lineal}", xlabel="log(1 + citas/año) real", ylabel="Predicho")
plt.tight_layout(); plt.savefig(FIG / "reg_comparacion.png"); plt.show()

# %% [markdown]
# ## Lasso como selección de variables

# %%
lasso = mejores["Lasso"]
nombres = lasso.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lasso.named_steps["reg"].coef_, index=[n.split("__", 1)[1] for n in nombres])
coef = coef[coef.abs() > 1e-6].sort_values()
print(f"Variables retenidas por Lasso: {len(coef)} de {len(nombres)}")
plt.figure(figsize=(8, 0.32 * len(coef) + 1))
plt.barh(coef.index, coef.values, color=np.where(coef.values > 0, "#55A868", "#C44E52"))
plt.axvline(0, c="black", lw=.8)
plt.title("Coeficientes de Lasso (variables estandarizadas)"); plt.xlabel("Efecto sobre log(1 + citas/año)")
plt.savefig(FIG / "reg_lasso_coeficientes.png"); plt.show()
coef.round(3)

# %%
res.to_csv(OUT / "regresion_resultados.csv", index=False)
coef.to_csv(OUT / "regresion_lasso_coeficientes.csv", header=["coeficiente"])
print("Guardado.")
