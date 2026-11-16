# Guía de despliegue en AWS (paso a paso)

Tiempo estimado: **45–60 minutos** la primera vez. Costo esperado: **centavos de dólar** (la base es serverless).
Los comandos están para **Windows (PowerShell)**; las diferencias para macOS/Linux se indican con 🐧.

---

## Fase 0 · Preparación (una sola vez)

### 0.1 Proteger la cuenta raíz
1. Inicia sesión en https://console.aws.amazon.com con el correo de tu cuenta (usuario raíz).
2. Arriba a la derecha → tu nombre → **Security credentials** → **Assign MFA device** → usa una app como Google Authenticator.
3. **Nunca** crees claves de acceso (access keys) para el usuario raíz.

### 0.2 Crear un usuario IAM para el proyecto
1. En el buscador de la consola escribe **IAM** → **Users** → **Create user**.
2. Nombre: `carlos-proyecto`. No marques acceso a la consola (no hace falta).
3. Permisos → **Attach policies directly** → marca **AdministratorAccess** (es una cuenta personal de estudio; en una empresa se darían permisos más acotados).
4. Crea el usuario → entra en él → pestaña **Security credentials** → **Create access key** → caso de uso **Command Line Interface (CLI)** → acepta → **Descarga el .csv**.
   ⚠️ Esas claves son como la contraseña de tu tarjeta: no las pegues en el código, en chats ni en GitHub.

### 0.3 Instalar las herramientas
Abre **PowerShell** y ejecuta:
```powershell
winget install -e --id Amazon.AWSCLI
winget install -e --id Hashicorp.Terraform
```
Cierra y vuelve a abrir PowerShell, luego verifica:
```powershell
aws --version
terraform -version
```
🐧 macOS: `brew install awscli terraform` · Linux: instaladores oficiales de AWS CLI v2 y HashiCorp.

### 0.4 Conectar la CLI con tu usuario
```powershell
aws configure
```
- AWS Access Key ID → del .csv
- AWS Secret Access Key → del .csv
- Default region name → `us-east-1`
- Default output format → `json`

Comprueba que funciona (debe mostrar tu número de cuenta y el usuario `carlos-proyecto`):
```powershell
aws sts get-caller-identity
```

---

## Fase A · Desplegar la infraestructura base

```powershell
cd C:\ruta\a\CarlosBarrera_codigo\terraform
copy terraform.tfvars.example terraform.tfvars
notepad terraform.tfvars
```
En el bloc de notas completa:
- `openalex_api_key` → tu clave de OpenAlex (del correo de activación).
- `email_alertas` → tu correo (recibirás alertas de presupuesto y errores).
- Deja `habilitar_dr = false` y `desplegar_api_inferencia = false` por ahora.

Luego:
```powershell
terraform init        # descarga el proveedor de AWS (una vez)
terraform plan        # muestra lo que va a crear; revisa que diga "Plan: N to add, 0 to change, 0 to destroy"
terraform apply       # escribe "yes" cuando lo pida
```
Al terminar verás las **salidas** (`bucket_datalake`, `lambda_ingesta`, etc.). Guárdalas en un bloc de notas.

📧 **Revisa tu correo**: llega un mensaje de *AWS Notifications* pidiendo confirmar la suscripción → haz clic en **Confirm subscription**.

---

## Fase B · Subir los datos y el modelo al data lake

Desde la carpeta `terraform` (la variable toma el nombre del bucket automáticamente):
```powershell
$B = terraform output -raw bucket_datalake
aws s3 cp ..\data\processed\imu_zancadas_limpio.csv  s3://$B/processed/imu_zancadas/imu_zancadas.csv
aws s3 cp ..\data\processed\articulos_limpio.csv     s3://$B/processed/articulos/articulos_limpio.csv
aws s3 cp ..\data\external\catalogo_datasets.csv     s3://$B/processed/catalogo/catalogo_datasets.csv
aws s3 cp ..\models\modelo_cojera.joblib             s3://$B/models/modelo_cojera.joblib
aws s3 ls s3://$B --recursive
```
🐧 macOS/Linux: `B=$(terraform output -raw bucket_datalake)` y rutas con `/`.

---

## Fase C · Probar que funciona

### C.1 Ejecutar la ingesta (scraping) a mano, sin esperar al lunes
```powershell
aws lambda invoke --function-name (terraform output -raw lambda_ingesta) --cli-read-timeout 700 respuesta.json
type respuesta.json
aws s3 ls s3://$B/raw/openalex/ --recursive
```
Debe mostrar `"statusCode": 200` y los registros por consulta, y aparecer 5 archivos `.jsonl` en S3.
(Si falla: consola → **Lambda** → la función → **Monitor** → **View CloudWatch logs**.)

### C.2 Consultar con SQL en Athena
1. Consola → **Athena** → **Query editor** → arriba a la derecha, Workgroup: `cojera-equina-dev-wg`.
2. Base de datos: `cojera_equina_dev_lake`. Ejecuta:
```sql
SELECT experimento, condicion, COUNT(*) AS zancadas, COUNT(DISTINCT id_caballo) AS caballos
FROM imu_zancadas
GROUP BY experimento, condicion
ORDER BY experimento, condicion;
```
Resultado esperado: 4 filas (anterior/posterior × BAS/IND) que suman 1.817 zancadas.

📸 **Toma captura** de: el resultado de `terraform apply`, el bucket S3 con sus carpetas, la Lambda ejecutada con éxito y la consulta de Athena. Son las evidencias para el informe y la sustentación.

---

## Fase D · (Opcional) API de inferencia en vivo

Requiere **Docker Desktop** instalado y abierto (https://www.docker.com/products/docker-desktop/).

```powershell
$ECR = terraform output -raw ecr_inferencia
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR.Split('/')[0]
docker build --platform linux/amd64 --provenance=false -t "${ECR}:latest" ..\src\cloud\lambda_inferencia
docker push "${ECR}:latest"
terraform apply -var "desplegar_api_inferencia=true"
```
Probar la API con 5 zancadas reales de un caballo cojo (`docs/ejemplo_prediccion.json`):
```powershell
$URL = terraform output -raw url_api
Invoke-RestMethod -Method Post -Uri $URL -ContentType "application/json" -InFile ..\docs\ejemplo_prediccion.json
```
🐧 `curl -X POST "$URL" -H "Content-Type: application/json" -d @../docs/ejemplo_prediccion.json`

Respuesta esperada: `probabilidad_cojera` alrededor de 0,6 y `cojo: true`. La predicción queda guardada en **DynamoDB** (consola → DynamoDB → Tables → `cojera-equina-dev-predicciones` → **Explore items**).

---

## Fase E · Costos reales (requisito R5)

1. Consola → **Billing and Cost Management** → **Cost allocation tags** → busca la etiqueta `Proyecto` → **Activate** (aparece hasta 24 h después del despliegue; es necesaria para filtrar costos y para el presupuesto).
2. Después de 1–3 días: **Cost Explorer** → filtro *Tag: Proyecto = deteccion-cojera-equina* → agrupar por *Service*. Toma captura: esos son los **costos reales** para comparar con la estimación.
3. **Budgets** → verás `cojera-equina-dev-presupuesto` con su límite de USD 10.

---

## Fase F · Apagar todo cuando termines

Cuando ya tengas las evidencias (o después de la sustentación):
```powershell
terraform destroy     # escribe "yes"
```
Borra todos los recursos del proyecto (incluidos los datos del bucket). Tu código y tus datos locales no se tocan; puedes volver a desplegar con `terraform apply` cuando quieras.

---

## Problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `Unable to locate credentials` | La CLI no está configurada | Repite `aws configure` |
| `AccessDenied` al hacer apply | El usuario no tiene permisos | Revisa que tenga `AdministratorAccess` |
| `BucketAlreadyExists` | Nombre de bucket repetido | Cambia `prefijo` en `terraform.tfvars` |
| La Lambda devuelve error 429 | Cupo de OpenAlex agotado | Verifica que `openalex_api_key` esté bien escrita y activada |
| `docker: command not found` | Docker no instalado o cerrado | Instala/abre Docker Desktop |
| Athena: `Table not found` | Base de datos mal seleccionada | Elige `cojera_equina_dev_lake` en el panel izquierdo |
