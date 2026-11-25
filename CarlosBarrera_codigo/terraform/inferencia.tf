# ---------------------------------------------------------------------------
# Inferencia (FaaS + API): API Gateway HTTP -> Lambda (imagen Docker con
# scikit-learn) que carga el modelo desde S3 y registra la predicción en DynamoDB.
# Se crea solo si desplegar_api_inferencia = true (requiere imagen en ECR).
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "inferencia" {
  name                 = "${local.nombre}-inferencia"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_lifecycle_policy" "inferencia" {
  repository = aws_ecr_repository.inferencia.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1, description = "Conservar solo 5 imágenes",
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 5 },
      action       = { type = "expire" }
    }]
  })
}

resource "aws_iam_role" "inferencia" {
  count              = var.desplegar_api_inferencia ? 1 : 0
  name               = "${local.nombre}-lambda-inferencia"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy" "inferencia" {
  count = var.desplegar_api_inferencia ? 1 : 0
  role  = aws_iam_role.inferencia[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:GetObject"], Resource = "${aws_s3_bucket.lake.arn}/models/*" },
      { Effect = "Allow", Action = ["dynamodb:PutItem"], Resource = aws_dynamodb_table.predicciones.arn },
      { Effect = "Allow", Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], Resource = "*" },
    ]
  })
}

resource "aws_lambda_function" "inferencia" {
  count         = var.desplegar_api_inferencia ? 1 : 0
  function_name = "${local.nombre}-inferencia"
  role          = aws_iam_role.inferencia[0].arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.inferencia.repository_url}:${var.imagen_inferencia_tag}"
  timeout       = 30
  memory_size   = 1024
  environment {
    variables = {
      BUCKET     = aws_s3_bucket.lake.id
      MODELO_KEY = "models/modelo_cojera.joblib"
      TABLA      = aws_dynamodb_table.predicciones.name
    }
  }
}

resource "aws_apigatewayv2_api" "api" {
  count         = var.desplegar_api_inferencia ? 1 : 0
  name          = "${local.nombre}-api"
  protocol_type = "HTTP"
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["POST", "GET"]
    allow_headers = ["content-type"]
  }
}

resource "aws_apigatewayv2_integration" "inferencia" {
  count                  = var.desplegar_api_inferencia ? 1 : 0
  api_id                 = aws_apigatewayv2_api.api[0].id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.inferencia[0].invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "predecir" {
  count     = var.desplegar_api_inferencia ? 1 : 0
  api_id    = aws_apigatewayv2_api.api[0].id
  route_key = "POST /predecir"
  target    = "integrations/${aws_apigatewayv2_integration.inferencia[0].id}"
}

resource "aws_apigatewayv2_stage" "default" {
  count       = var.desplegar_api_inferencia ? 1 : 0
  api_id      = aws_apigatewayv2_api.api[0].id
  name        = "$default"
  auto_deploy = true
  default_route_settings {
    throttling_burst_limit = 50 # protege costos ante abuso
    throttling_rate_limit  = 20
  }
}

resource "aws_lambda_permission" "apigw" {
  count         = var.desplegar_api_inferencia ? 1 : 0
  statement_id  = "PermitirAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.inferencia[0].function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api[0].execution_arn}/*/*"
}
