terraform {
  required_version = ">= 1.6"
  required_providers {
    aws     = { source = "hashicorp/aws", version = "~> 5.70" }
    archive = { source = "hashicorp/archive", version = "~> 2.6" }
  }
  # Estado remoto recomendado para trabajo en equipo (descomentar tras crear el bucket y la tabla de bloqueo):
  # backend "s3" {
  #   bucket         = "cojera-equina-tfstate"
  #   key            = "infra/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "cojera-equina-tflock"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Proyecto      = "deteccion-cojera-equina"
      Asignaturas   = "MineriaDatos-MachineLearning-CloudComputing"
      Autor         = "Carlos Barrera"
      Entorno       = var.entorno
      ODS           = "3-8"
      GestionadoPor = "terraform"
    }
  }
}

# Proveedor secundario para recuperación ante desastres (replicación entre regiones)
provider "aws" {
  alias  = "dr"
  region = var.region_dr
}
