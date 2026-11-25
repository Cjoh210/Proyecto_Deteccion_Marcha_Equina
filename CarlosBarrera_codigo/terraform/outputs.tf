output "bucket_datalake" {
  description = "Bucket S3 del data lake"
  value       = aws_s3_bucket.lake.id
}

output "lambda_ingesta" {
  description = "Función de ingesta (web scraping OpenAlex)"
  value       = aws_lambda_function.ingesta.function_name
}

output "athena_workgroup" {
  value = aws_athena_workgroup.lake.name
}

output "tabla_predicciones" {
  value = aws_dynamodb_table.predicciones.name
}

output "ecr_inferencia" {
  description = "Repositorio donde subir la imagen Docker de inferencia"
  value       = aws_ecr_repository.inferencia.repository_url
}

output "url_api" {
  description = "Endpoint POST /predecir (si se desplegó la API)"
  value       = var.desplegar_api_inferencia ? "${aws_apigatewayv2_api.api[0].api_endpoint}/predecir" : "no desplegada"
}
