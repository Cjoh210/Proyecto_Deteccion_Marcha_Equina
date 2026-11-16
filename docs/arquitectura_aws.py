"""Genera el diagrama de arquitectura AWS (docs/figures/arquitectura_aws.png) con la librería `diagrams`.
Uso: python docs/arquitectura_aws.py
"""
from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.analytics import Athena, GlueDataCatalog
from diagrams.aws.compute import ECR, Lambda
from diagrams.aws.database import Dynamodb
from diagrams.aws.integration import Eventbridge, SNS
from diagrams.aws.management import Cloudwatch
from diagrams.aws.ml import Sagemaker
from diagrams.aws.network import APIGateway
from diagrams.aws.management import SystemsManagerParameterStore
from diagrams.aws.storage import S3
from diagrams.onprem.client import Users
from diagrams.programming.language import Python
from diagrams.generic.storage import Storage

OUT = Path(__file__).resolve().parent / "figures" / "arquitectura_aws"
graph_attr = {"fontsize": "20", "pad": "0.4", "splines": "spline", "nodesep": "0.6", "ranksep": "0.9"}

with Diagram("Sistema de detección de cojera equina · AWS (us-east-1)", filename=str(OUT), show=False,
             direction="LR", graph_attr=graph_attr, outformat="png"):
    openalex = Python("API OpenAlex\n(web scraping)")
    dataverse = Storage("DataverseNL\n(dataset IMU)")
    usuario = Users("Veterinario /\ncriador")

    with Cluster("1 · Ingesta (FaaS)"):
        cron = Eventbridge("EventBridge\nsemanal")
        ingesta = Lambda("Lambda\ningesta")
        secreto = SystemsManagerParameterStore("SSM Parameter\nStore")

    with Cluster("2 · Data lake (S3, versionado + ciclo de vida)"):
        raw = S3("raw/")
        processed = S3("processed/")
        models = S3("models/")

    with Cluster("3 · Minería y ML"):
        catalogo = GlueDataCatalog("Glue Data\nCatalog")
        athena = Athena("Athena (SQL)")
        sm = Sagemaker("SageMaker / notebooks\n(KDD + entrenamiento)")

    with Cluster("4 · Servicio de inferencia (serverless)"):
        api = APIGateway("API Gateway\nHTTP")
        infer = Lambda("Lambda\n(Docker, sklearn)")
        ecr = ECR("ECR")
        ddb = Dynamodb("DynamoDB\npredicciones (PITR)")

    with Cluster("Operación"):
        cw = Cloudwatch("CloudWatch\n+ Budgets")
        sns = SNS("SNS alertas")

    dr = S3("Réplica DR\n(us-west-2)")

    cron >> ingesta
    secreto >> Edge(style="dashed") >> ingesta
    openalex >> ingesta >> raw
    dataverse >> Edge(label="descarga") >> raw
    raw >> Edge(label="limpieza / features") >> processed
    processed >> catalogo >> athena
    processed >> sm >> Edge(label="modelo .joblib") >> models
    usuario >> Edge(label="POST /predecir") >> api >> infer
    ecr >> Edge(style="dashed") >> infer
    models >> infer >> ddb
    processed >> Edge(style="dotted", label="CRR") >> dr
    ingesta >> Edge(style="dotted") >> cw >> sns
