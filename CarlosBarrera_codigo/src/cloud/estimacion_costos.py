"""
Estimación preliminar de costos mensuales de la arquitectura AWS (us-east-1).

Dos escenarios:
  * Académico (avance): datos del proyecto, ingesta semanal, ~1.000 predicciones de prueba/mes.
  * Piloto (proyección): 50 clínicas/criaderos, 20.000 pruebas de trote/mes, réplica DR activa.

Precios de lista on-demand en us-east-1 consultados en noviembre de 2026. Los que vienen de las páginas
de precios de AWS o de guías verificadas están marcados en FUENTES. Antes de desplegar, confirmar en
https://calculator.aws (los precios cambian). Se reporta el costo SIN y CON capa gratuita.

Uso: python src/cloud/estimacion_costos.py   ->  data/processed/costos_aws.csv
"""
from pathlib import Path

import pandas as pd

P = {  # USD
    "lambda_gb_s_arm": 0.0000133334, "lambda_gb_s_x86": 0.0000166667, "lambda_req_m": 0.20,
    "apigw_http_req_m": 1.00, "ddb_wru_m": 0.625, "ddb_rru_m": 0.125, "ddb_gb_mes": 0.25, "ddb_pitr_gb_mes": 0.20,
    "s3_std_gb_mes": 0.023, "s3_ia_gb_mes": 0.0125, "s3_glacier_gb_mes": 0.0036, "s3_put_k": 0.005,
    "s3_transfer_interregion_gb": 0.02, "athena_tb": 5.0, "secrets_mes": 0.40, "ecr_gb_mes": 0.10,
    "cw_logs_ingesta_gb": 0.50,
}
FUENTES = {
    "Lambda": "aws.amazon.com/lambda/pricing ($0.20/M solicitudes; $0.0000166667 GB-s x86; $0.0000133334 GB-s arm)",
    "API Gateway": "aws.amazon.com/api-gateway/pricing (HTTP API $1.00/M)",
    "DynamoDB": "Precio on-demand tras la rebaja de nov-2024 ($0.625/M WRU, $0.125/M RRU)",
    "S3/Athena/ECR/CloudWatch": "Precios de lista us-east-1 (verificar en calculator.aws)",
}
FREE = {"lambda_gb_s": 400_000, "lambda_req": 1_000_000, "apigw_req": 1_000_000, "ddb_gb": 25,
        "cw_logs_gb": 5, "s3_gb": 5}

ESCENARIOS = {
    "Académico (avance)": dict(s3_std=7.0, s3_glacier=0, s3_put=2_000, ingestas=4, ingesta_s=300, ingesta_gb=0.5,
                               pred=1_000, pred_s=1.0, pred_gb=1.0, ddb_gb=0.01, athena_q=100, athena_mb=50,
                               ecr_gb=0.5, logs_gb=0.2, dr_gb=0, dr_nuevo_gb=0),
    "Piloto (50 clínicas)": dict(s3_std=200.0, s3_glacier=500, s3_put=100_000, ingestas=4, ingesta_s=300, ingesta_gb=0.5,
                                 pred=20_000, pred_s=2.0, pred_gb=1.0, ddb_gb=1.0, athena_q=1_000, athena_mb=200,
                                 ecr_gb=0.5, logs_gb=3.0, dr_gb=200, dr_nuevo_gb=20),
}


def costos(e, capa_gratuita):
    f = FREE if capa_gratuita else {k: 0 for k in FREE}
    filas = []
    def add(servicio, componente, usd, supuesto):
        filas.append({"servicio": servicio, "componente": componente, "usd_mes": max(usd, 0), "supuesto": supuesto})

    gb_s_ing = e["ingestas"] * e["ingesta_s"] * e["ingesta_gb"]
    gb_s_pred = e["pred"] * e["pred_s"] * e["pred_gb"]
    gb_libre_ing = min(gb_s_ing, f["lambda_gb_s"])
    gb_libre_pred = max(min(gb_s_pred, f["lambda_gb_s"] - gb_libre_ing), 0)
    req_total = e["ingestas"] + e["pred"]
    add("Lambda", "Ingesta OpenAlex (arm64)", (gb_s_ing - gb_libre_ing) * P["lambda_gb_s_arm"],
        f"{e['ingestas']} ejecuciones × {e['ingesta_s']} s × {e['ingesta_gb']} GB")
    add("Lambda", "Inferencia (x86, Docker)", (gb_s_pred - gb_libre_pred) * P["lambda_gb_s_x86"],
        f"{e['pred']:,} pruebas × {e['pred_s']} s × {e['pred_gb']} GB")
    add("Lambda", "Solicitudes", max(req_total - f["lambda_req"], 0) / 1e6 * P["lambda_req_m"], f"{req_total:,} solicitudes")
    add("API Gateway", "HTTP API", max(e["pred"] - f["apigw_req"], 0) / 1e6 * P["apigw_http_req_m"], f"{e['pred']:,} solicitudes")
    add("DynamoDB", "Escrituras on-demand + PITR", e["pred"] / 1e6 * P["ddb_wru_m"] + e["ddb_gb"] * P["ddb_pitr_gb_mes"]
        + max(e["ddb_gb"] - f["ddb_gb"], 0) * P["ddb_gb_mes"], f"{e['pred']:,} escrituras, {e['ddb_gb']} GB")
    add("S3", "Standard (data lake)", max(e["s3_std"] - f["s3_gb"], 0) * P["s3_std_gb_mes"], f"{e['s3_std']} GB")
    add("S3", "Glacier (raw > 90 días)", e["s3_glacier"] * P["s3_glacier_gb_mes"], f"{e['s3_glacier']} GB")
    add("S3", "Solicitudes PUT", e["s3_put"] / 1000 * P["s3_put_k"], f"{e['s3_put']:,} PUT")
    add("S3 (DR)", "Réplica Standard-IA + transferencia", e["dr_gb"] * P["s3_ia_gb_mes"] + e["dr_nuevo_gb"] * P["s3_transfer_interregion_gb"],
        f"{e['dr_gb']} GB almacenados en us-west-2; {e['dr_nuevo_gb']} GB nuevos replicados/mes")
    add("Athena", "Consultas SQL", e["athena_q"] * max(e["athena_mb"], 10) / 1e6 * P["athena_tb"],
        f"{e['athena_q']} consultas × {e['athena_mb']} MB")
    add("SSM Parameter Store", "Clave OpenAlex (SecureString estándar)", 0.0, "Nivel estándar sin costo")
    add("ECR", "Imagen de inferencia", e["ecr_gb"] * P["ecr_gb_mes"], f"{e['ecr_gb']} GB")
    add("CloudWatch", "Logs", max(e["logs_gb"] - f["cw_logs_gb"], 0) * P["cw_logs_ingesta_gb"], f"{e['logs_gb']} GB")
    add("Glue / Budgets / SNS / EventBridge", "Catálogo, alertas, programación", 0.0,
        "Dentro de cuotas gratuitas (1 M objetos en catálogo, 2 presupuestos, reglas programadas)")
    return pd.DataFrame(filas)


def main():
    out = []
    for nombre, e in ESCENARIOS.items():
        for fg in (False, True):
            d = costos(e, fg); d["escenario"] = nombre; d["capa_gratuita"] = fg; out.append(d)
    df = pd.concat(out)
    root = Path(__file__).resolve().parents[2]
    df.to_csv(root / "data" / "processed" / "costos_aws.csv", index=False)
    tabla = df.pivot_table(index=["servicio", "componente"], columns=["escenario", "capa_gratuita"], values="usd_mes", aggfunc="sum")
    print(tabla.round(3).to_string())
    print("\nTOTAL mensual (USD):")
    print(df.groupby(["escenario", "capa_gratuita"])["usd_mes"].sum().round(2).to_string())


if __name__ == "__main__":
    main()
