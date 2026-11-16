# ---------------------------------------------------------------------------
# Ingesta (FaaS): Lambda programada que consulta la API de OpenAlex y guarda
# el JSON crudo en s3://<lake>/raw/openalex/ (módulo de web scraping del sistema)
# ---------------------------------------------------------------------------
# Clave de OpenAlex en SSM Parameter Store (SecureString, nivel estándar: sin costo,
# cifrada con la clave administrada aws/ssm). Alternativa gratuita a Secrets Manager.
resource "aws_ssm_parameter" "openalex" {
  name        = "/${local.nombre}/openalex-api-key"
  description = "Clave de la API de OpenAlex para la Lambda de ingesta"
  type        = "SecureString"
  tier        = "Standard"
  value       = var.openalex_api_key == "" ? "SIN_CLAVE" : var.openalex_api_key
}

data "archive_file" "ingesta" {
  type        = "zip"
  source_dir  = "${path.module}/../src/cloud/lambda_ingesta"
  output_path = "${path.module}/build/lambda_ingesta.zip"
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ingesta" {
  name               = "${local.nombre}-lambda-ingesta"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

# Mínimo privilegio: solo escribe en raw/openalex/, lee su parámetro y escribe logs
resource "aws_iam_role_policy" "ingesta" {
  role = aws_iam_role.ingesta.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:PutObject"], Resource = "${aws_s3_bucket.lake.arn}/raw/openalex/*" },
      { Effect = "Allow", Action = ["ssm:GetParameter"], Resource = aws_ssm_parameter.openalex.arn },
      { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "${aws_cloudwatch_log_group.ingesta.arn}:*" },
    ]
  })
}

resource "aws_cloudwatch_log_group" "ingesta" {
  name              = "/aws/lambda/${local.nombre}-ingesta-openalex"
  retention_in_days = 14
}

resource "aws_lambda_function" "ingesta" {
  function_name    = "${local.nombre}-ingesta-openalex"
  role             = aws_iam_role.ingesta.arn
  runtime          = "python3.12"
  handler          = "handler.lambda_handler"
  filename         = data.archive_file.ingesta.output_path
  source_code_hash = data.archive_file.ingesta.output_base64sha256
  timeout          = 600
  memory_size      = 256
  architectures    = ["arm64"] # Graviton: ~20 % más barato por GB-segundo
  environment {
    variables = {
      BUCKET           = aws_s3_bucket.lake.id
      PREFIJO          = "raw/openalex/"
      PARAM_CLAVE      = aws_ssm_parameter.openalex.name
      MAX_POR_CONSULTA = "1500"
    }
  }
  depends_on = [aws_cloudwatch_log_group.ingesta]
}

resource "aws_cloudwatch_event_rule" "ingesta" {
  name                = "${local.nombre}-ingesta-semanal"
  schedule_expression = var.horario_ingesta
}

resource "aws_cloudwatch_event_target" "ingesta" {
  rule = aws_cloudwatch_event_rule.ingesta.name
  arn  = aws_lambda_function.ingesta.arn
}

resource "aws_lambda_permission" "eventbridge" {
  statement_id  = "PermitirEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingesta.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.ingesta.arn
}
