"""English version of the AWS architecture diagram (report/final/en/figures/arquitectura_aws.png).
Usage: python docs/arquitectura_aws_en.py
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

OUT = Path(__file__).resolve().parents[1] / "report" / "final" / "en" / "figures" / "arquitectura_aws"
graph_attr = {"fontsize": "20", "pad": "0.4", "splines": "spline", "nodesep": "0.6", "ranksep": "0.9"}

with Diagram("Equine lameness detection system · AWS (us-east-1)", filename=str(OUT), show=False,
             direction="LR", graph_attr=graph_attr, outformat="png"):
    openalex = Python("API OpenAlex\n(web scraping)")
    dataverse = Storage("DataverseNL\n(IMU dataset)")
    usuario = Users("Veterinarian /\nbreeder")

    with Cluster("1 · Ingestion (FaaS)"):
        cron = Eventbridge("EventBridge\nweekly")
        ingesta = Lambda("Lambda\ningestion")
        secreto = SystemsManagerParameterStore("SSM Parameter\nStore")

    with Cluster("2 · Data lake (S3, versioning + lifecycle)"):
        raw = S3("raw/")
        processed = S3("processed/")
        models = S3("models/")

    with Cluster("3 · Data mining and ML"):
        catalogo = GlueDataCatalog("Glue Data\nCatalog")
        athena = Athena("Athena (SQL)")
        sm = Sagemaker("SageMaker / notebooks\n(KDD + training)")

    with Cluster("4 · Inference service (serverless)"):
        api = APIGateway("API Gateway\nHTTP")
        infer = Lambda("Lambda\n(Docker, sklearn)")
        ecr = ECR("ECR")
        ddb = Dynamodb("DynamoDB\npredictions (PITR)")

    with Cluster("Operations"):
        cw = Cloudwatch("CloudWatch\n+ Budgets")
        sns = SNS("SNS alerts")

    dr = S3("DR replica\n(us-west-2)")

    cron >> ingesta
    secreto >> Edge(style="dashed") >> ingesta
    openalex >> ingesta >> raw
    dataverse >> Edge(label="download") >> raw
    raw >> Edge(label="cleaning / features") >> processed
    processed >> catalogo >> athena
    processed >> sm >> Edge(label="model .joblib") >> models
    usuario >> Edge(label="POST /predecir") >> api >> infer
    ecr >> Edge(style="dashed") >> infer
    models >> infer >> ddb
    processed >> Edge(style="dotted", label="CRR") >> dr
    ingesta >> Edge(style="dotted") >> cw >> sns
