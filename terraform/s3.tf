# ---------------------------------------------------------------------------
# Almacenamiento: data lake en S3 con zonas raw / processed / models
# ---------------------------------------------------------------------------
data "aws_caller_identity" "actual" {}

locals {
  nombre      = "${var.prefijo}-${var.entorno}"
  bucket_lake = "${local.nombre}-datalake-${data.aws_caller_identity.actual.account_id}"
}

resource "aws_s3_bucket" "lake" {
  bucket        = local.bucket_lake
  force_destroy = var.entorno != "prod"
}

resource "aws_s3_bucket_versioning" "lake" {
  bucket = aws_s3_bucket.lake.id
  versioning_configuration { status = "Enabled" } # protege contra borrados accidentales (DR)
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lake" {
  bucket = aws_s3_bucket.lake.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "lake" {
  bucket                  = aws_s3_bucket.lake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Ciclo de vida: los datos crudos (señales IMU, JSON de la API) pasan a Glacier a los 90 días
resource "aws_s3_bucket_lifecycle_configuration" "lake" {
  bucket = aws_s3_bucket.lake.id
  rule {
    id     = "raw-a-glacier"
    status = "Enabled"
    filter { prefix = "raw/" }
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    noncurrent_version_expiration { noncurrent_days = 180 }
  }
  rule {
    id     = "athena-resultados-temporales"
    status = "Enabled"
    filter { prefix = "athena-results/" }
    expiration { days = 30 }
  }
}

# Prefijos (zonas) del data lake
resource "aws_s3_object" "zonas" {
  for_each = toset(["raw/openalex/", "raw/imu/", "processed/", "models/", "athena-results/"])
  bucket   = aws_s3_bucket.lake.id
  key      = each.value
  content  = ""
}

# ---------------------------------------------------------------------------
# Recuperación ante desastres (opcional): réplica entre regiones
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "replica" {
  count         = var.habilitar_dr ? 1 : 0
  provider      = aws.dr
  bucket        = "${local.bucket_lake}-replica"
  force_destroy = var.entorno != "prod"
}

resource "aws_s3_bucket_versioning" "replica" {
  count    = var.habilitar_dr ? 1 : 0
  provider = aws.dr
  bucket   = aws_s3_bucket.replica[0].id
  versioning_configuration { status = "Enabled" }
}

data "aws_iam_policy_document" "replicacion_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["s3.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "replicacion" {
  count              = var.habilitar_dr ? 1 : 0
  name               = "${local.nombre}-s3-replicacion"
  assume_role_policy = data.aws_iam_policy_document.replicacion_assume.json
}

resource "aws_iam_role_policy" "replicacion" {
  count = var.habilitar_dr ? 1 : 0
  role  = aws_iam_role.replicacion[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:GetReplicationConfiguration", "s3:ListBucket"], Resource = aws_s3_bucket.lake.arn },
      { Effect = "Allow", Action = ["s3:GetObjectVersionForReplication", "s3:GetObjectVersionAcl", "s3:GetObjectVersionTagging"], Resource = "${aws_s3_bucket.lake.arn}/*" },
      { Effect = "Allow", Action = ["s3:ReplicateObject", "s3:ReplicateDelete", "s3:ReplicateTags"], Resource = "${aws_s3_bucket.replica[0].arn}/*" },
    ]
  })
}

resource "aws_s3_bucket_replication_configuration" "lake" {
  count      = var.habilitar_dr ? 1 : 0
  depends_on = [aws_s3_bucket_versioning.lake, aws_s3_bucket_versioning.replica]
  bucket     = aws_s3_bucket.lake.id
  role       = aws_iam_role.replicacion[0].arn
  rule {
    id     = "processed-y-models"
    status = "Enabled"
    filter {}
    delete_marker_replication { status = "Enabled" }
    destination {
      bucket        = aws_s3_bucket.replica[0].arn
      storage_class = "STANDARD_IA"
    }
  }
}
