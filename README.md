# 🔬 VisionAI — Clasificador de Imágenes en Microservicios

Arquitectura de microservicios desplegada en **Amazon EKS** para clasificación
automática de imágenes usando **MobileNetV2** (ImageNet).  
Práctica 5 — Computación en la Nube · Universidad Autónoma de Occidente

---

## 📐 Arquitectura

```
Internet
   │
   ▼
[AWS ALB / LoadBalancer]
   │  puerto 80
   ▼
[frontend :5000]  ── Flask + HTML/CSS/JS
   │
   ├──► [ai-service :5001]   Flask + TensorFlow MobileNetV2
   │
   └──► [data-service :5002] Flask + psycopg2
             │
             ▼
         [PostgreSQL :5432]  PVC en EKS (gp2)
```

### Microservicios

| Servicio | Puerto | Tecnología | Función |
|---|---|---|---|
| `frontend` | 5000 | Flask + HTML/CSS/JS | Dashboard UI, proxy entre usuario y servicios |
| `ai-service` | 5001 | Flask + TensorFlow CPU | Clasificación con MobileNetV2 (top-5 ImageNet) |
| `data-service` | 5002 | Flask + psycopg2 | CRUD sobre PostgreSQL (historial, estadísticas) |
| `postgres` | 5432 | PostgreSQL 16 | Almacenamiento persistente con PVC |

---

## 🔧 Pre-requisitos

- AWS CLI configurado (`aws configure`)
- `eksctl` instalado
- `kubectl` instalado
- `docker` instalado
- Cuenta AWS con permisos sobre EKS, ECR, IAM, VPC

---

## 🚀 Paso a Paso

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu-usuario>/visionai-eks.git
cd visionai-eks
```

---

### 2. Crear repositorios en Amazon ECR

```bash
export AWS_REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

for SERVICE in frontend ai-service data-service; do
  aws ecr create-repository \
    --repository-name $SERVICE \
    --region $AWS_REGION
done
```

---

### 3. Autenticarse en ECR y construir las imágenes

```bash
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS \
    --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

```bash
# Frontend
docker build -t frontend ./frontend
docker tag frontend:latest \
  $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/frontend:latest
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/frontend:latest

# AI Service (más lento — descarga pesos de TensorFlow)
docker build -t ai-service ./ai-service
docker tag ai-service:latest \
  $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ai-service:latest
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ai-service:latest

# Data Service
docker build -t data-service ./data-service
docker tag data-service:latest \
  $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/data-service:latest
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/data-service:latest
```

---

### 4. Crear el clúster EKS

```bash
eksctl create cluster \
  --name visionai-cluster \
  --region $AWS_REGION \
  --nodegroup-name standard-nodes \
  --node-type t3.large \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 4 \
  --managed
```

> ⏳ Este proceso tarda ~15 minutos.

Verificar que los nodos estén listos:

```bash
kubectl get nodes
```

---

### 5. Actualizar las imágenes en los manifiestos

Reemplaza `<ACCOUNT_ID>` y `<REGION>` en los archivos YAML:

```bash
# Linux / Mac
sed -i "s/<ACCOUNT_ID>/$ACCOUNT_ID/g" k8s/ai-service.yaml k8s/data-service.yaml k8s/frontend.yaml
sed -i "s/<REGION>/$AWS_REGION/g"     k8s/ai-service.yaml k8s/data-service.yaml k8s/frontend.yaml

# Windows PowerShell
$files = "k8s/ai-service.yaml","k8s/data-service.yaml","k8s/frontend.yaml"
foreach ($f in $files) {
  (Get-Content $f) -replace '<ACCOUNT_ID>',$env:ACCOUNT_ID `
                   -replace '<REGION>',$env:AWS_REGION | Set-Content $f
}
```

---

### 6. Desplegar en EKS

```bash
# Secret con credenciales de Postgres
kubectl apply -f k8s/postgres-secret.yaml

# Base de datos
kubectl apply -f k8s/postgres.yaml

# Esperar a que Postgres esté listo
kubectl wait --for=condition=ready pod -l app=postgres --timeout=120s

# Microservicios
kubectl apply -f k8s/data-service.yaml
kubectl apply -f k8s/ai-service.yaml
kubectl apply -f k8s/frontend.yaml
```

---

### 7. Obtener la URL pública

```bash
kubectl get svc frontend
```

Busca la columna `EXTERNAL-IP`. Puede tardar 2-3 minutos en asignarse.

```
NAME       TYPE           CLUSTER-IP    EXTERNAL-IP                          PORT(S)
frontend   LoadBalancer   10.100.x.x    abc123.us-east-1.elb.amazonaws.com   80:xxxxx/TCP
```

Abre `http://<EXTERNAL-IP>` en tu navegador. 🎉

---

### 8. Verificar el estado del clúster

```bash
# Ver todos los pods
kubectl get pods

# Ver logs de un servicio
kubectl logs -l app=ai-service --tail=50
kubectl logs -l app=data-service --tail=50
kubectl logs -l app=frontend --tail=50

# Describir un pod con error
kubectl describe pod <nombre-del-pod>
```

---

## 🖼️ Funcionalidades de la Aplicación

| Feature | Descripción |
|---|---|
| **Drag & Drop** | Arrastra una imagen o haz clic para seleccionar |
| **Clasificación en tiempo real** | Top-5 categorías con barras de confianza animadas |
| **Guardado automático** | Cada clasificación se persiste en PostgreSQL |
| **Historial paginado** | Galería de imágenes clasificadas con thumbnails |
| **Re-clasificar** | Vuelve a correr el modelo sobre una imagen anterior |
| **Eliminar registros** | Borra clasificaciones del historial y la BD |
| **Modal de detalles** | Clic en una imagen del historial para ver predicciones completas |
| **Dashboard de estadísticas** | Total, confianza promedio, top categorías, actividad diaria |
| **Gráficas interactivas** | Chart.js — barras horizontales + línea de actividad |

---

## 🧹 Limpieza (para no generar costos)

```bash
# Eliminar todos los recursos de K8s
kubectl delete -f k8s/

# Eliminar el clúster EKS
eksctl delete cluster --name visionai-cluster --region $AWS_REGION

# Eliminar repositorios ECR
for SERVICE in frontend ai-service data-service; do
  aws ecr delete-repository \
    --repository-name $SERVICE \
    --force \
    --region $AWS_REGION
done
```

---

## 📁 Estructura del Proyecto

```
visionai-eks/
├── frontend/
│   ├── app.py                # Flask proxy
│   ├── templates/
│   │   └── index.html        # UI (dark theme, drag-drop, charts)
│   ├── requirements.txt
│   └── Dockerfile
├── ai-service/
│   ├── app.py                # MobileNetV2 classifier
│   ├── requirements.txt
│   └── Dockerfile
├── data-service/
│   ├── app.py                # PostgreSQL CRUD
│   ├── requirements.txt
│   └── Dockerfile
├── k8s/
│   ├── postgres-secret.yaml
│   ├── postgres.yaml          # PVC + Deployment + Service
│   ├── ai-service.yaml        # Deployment + Service (ClusterIP)
│   ├── data-service.yaml      # Deployment + Service (ClusterIP)
│   └── frontend.yaml          # Deployment + Service (LoadBalancer)
└── README.md
```

---

## 👤 Autor

**Adrian Felipe Vargas Rojas**  
Computación en la Nube — Universidad Autónoma de Occidente  
Docente: Jhorman A. Villanueva Vivas
