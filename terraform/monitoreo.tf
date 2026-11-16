# ---------------------------------------------------------------------------
# Monitoreo y control de costos
# ---------------------------------------------------------------------------
resource "aws_sns_topic" "alertas" {
  name = "${local.nombre}-alertas"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.email_alertas == "" ? 0 : 1
  topic_arn = aws_sns_topic.alertas.arn
  protocol  = "email"
  endpoint  = var.email_alertas
}

resource "aws_cloudwatch_metric_alarm" "errores_ingesta" {
  alarm_name          = "${local.nombre}-errores-ingesta"
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.ingesta.function_name }
  statistic           = "Sum"
  period              = 3600
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alertas.arn]
}

resource "aws_budgets_budget" "mensual" {
  name         = "${local.nombre}-presupuesto"
  budget_type  = "COST"
  limit_amount = tostring(var.presupuesto_mensual_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  cost_filter {
    name   = "TagKeyValue"
    values = ["user:Proyecto$deteccion-cojera-equina"]
  }
  dynamic "notification" {
    for_each = var.email_alertas == "" ? [] : [80, 100]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = notification.value == 100 ? "FORECASTED" : "ACTUAL"
      subscriber_email_addresses = [var.email_alertas]
    }
  }
}
