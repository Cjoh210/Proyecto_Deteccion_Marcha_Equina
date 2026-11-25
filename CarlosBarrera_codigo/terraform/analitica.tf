# ---------------------------------------------------------------------------
# Analítica (PaaS/serverless): Glue Data Catalog + Athena para consultar con SQL
# los datasets limpios (artículos, zancadas IMU, catálogo) directamente sobre S3
# ---------------------------------------------------------------------------
resource "aws_glue_catalog_database" "lake" {
  name = replace("${local.nombre}_lake", "-", "_")
}

resource "aws_glue_catalog_table" "zancadas" {
  database_name = aws_glue_catalog_database.lake.name
  name          = "imu_zancadas"
  table_type    = "EXTERNAL_TABLE"
  parameters    = { "skip.header.line.count" = "1", classification = "csv" }
  storage_descriptor {
    location      = "s3://${aws_s3_bucket.lake.id}/processed/imu_zancadas/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"
      parameters            = { "field.delim" = "," }
    }
    columns {
      name = "experimento"
      type = "string"
    }
    columns {
      name = "caballo"
      type = "string"
    }
    columns {
      name = "id_caballo"
      type = "string"
    }
    columns {
      name = "condicion"
      type = "string"
    }
    columns {
      name = "cojo"
      type = "int"
    }
    # El resto de columnas (métricas por zancada) se registra con el crawler o se amplía aquí.
  }
}

resource "aws_athena_workgroup" "lake" {
  name          = "${local.nombre}-wg"
  force_destroy = true
  configuration {
    enforce_workgroup_configuration    = true
    bytes_scanned_cutoff_per_query     = 1073741824 # 1 GB por consulta: control de costos
    publish_cloudwatch_metrics_enabled = true
    result_configuration {
      output_location = "s3://${aws_s3_bucket.lake.id}/athena-results/"
    }
  }
}

# ---------------------------------------------------------------------------
# Base de datos operacional: DynamoDB (registro de predicciones por caballo)
# On-demand: escala automáticamente y es Multi-AZ por diseño.
# ---------------------------------------------------------------------------
resource "aws_dynamodb_table" "predicciones" {
  name         = "${local.nombre}-predicciones"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id_caballo"
  range_key    = "timestamp"

  attribute {
    name = "id_caballo"
    type = "S"
  }
  attribute {
    name = "timestamp"
    type = "S"
  }

  point_in_time_recovery { enabled = true } # DR: restauración a cualquier segundo de los últimos 35 días
  server_side_encryption { enabled = true }
}
