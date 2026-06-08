# =============================================================
# Deploy manual: rebuild imagen ECR + actualizar Lambda t2
# Ejecutar desde la raíz del repo
# =============================================================

set -euo pipefail

AWS_REGION="eu-south-2"
AWS_ACCOUNT="610140802215"
ECR_REPO="t2-inference-lambda"
LAMBDA_NAME="t2-inference-lambda"
IMAGE_URI="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "▶ Autenticando en ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "▶ Build y push a ECR (linux/amd64, sin manifest list)..."
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --push \
  -t "${IMAGE_URI}:latest" .

echo "▶ Resolviendo digest exacto del manifest pusheado..."
DIGEST=$(aws ecr describe-images \
  --repository-name "$ECR_REPO" \
  --region "$AWS_REGION" \
  --image-ids imageTag=latest \
  --query 'imageDetails[0].imageDigest' \
  --output text)

echo "   Digest: ${DIGEST}"

echo "▶ Actualizando Lambda con digest (bypass tag resolution)..."
aws lambda update-function-code \
  --region "$AWS_REGION" \
  --function-name "$LAMBDA_NAME" \
  --image-uri "${IMAGE_URI}@${DIGEST}"

echo "▶ Esperando a que Lambda termine de actualizarse..."
aws lambda wait function-updated \
  --region "$AWS_REGION" \
  --function-name "$LAMBDA_NAME"

echo "✅ Deploy completado: ${IMAGE_URI}@${DIGEST} → ${LAMBDA_NAME}"



