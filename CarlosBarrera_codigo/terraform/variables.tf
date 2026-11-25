variable "region" {
  description = "Región principal de AWS"
  type        = string
  default     = "us-east-1"
}

variable "region_dr" {
  description = "Región secundaria para recuperación ante desastres"
  type        = string
  default     = "us-west-2"
}

variable "entorno" {
  description = "Nombre del entorno (dev, prod)"
  type        = string
  default     = "dev"
}

variable "prefijo" {
  description = "Prefijo para nombrar recursos (minúsculas, sin espacios)"
  type        = string
  default     = "cojera-equina"
}

variable "habilitar_dr" {
  description = "Crea un bucket réplica en otra región (recuperación ante desastres)"
  type        = bool
  default     = false
}

variable "desplegar_api_inferencia" {
  description = "Despliega la API de inferencia (requiere haber subido la imagen a ECR)"
  type        = bool
  default     = false
}

variable "imagen_inferencia_tag" {
  description = "Tag de la imagen Docker de inferencia en ECR"
  type        = string
  default     = "latest"
}

variable "openalex_api_key" {
  description = "Clave gratuita de OpenAlex (se guarda cifrada en SSM Parameter Store; no la versiones en git)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "horario_ingesta" {
  description = "Expresión de EventBridge para la ingesta periódica de artículos"
  type        = string
  default     = "cron(0 6 ? * MON *)" # lunes 06:00 UTC
}

variable "presupuesto_mensual_usd" {
  description = "Tope de presupuesto mensual para la alerta de costos"
  type        = number
  default     = 10
}

variable "email_alertas" {
  description = "Correo para alertas de presupuesto y errores (vacío = sin suscripción)"
  type        = string
  default     = ""
}
