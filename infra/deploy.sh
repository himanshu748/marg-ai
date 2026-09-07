#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="${STACK_NAME:-margai}"
REPOSITORY="${ECR_REPOSITORY:-margai}"
MODEL_ID="${MARG_BEDROCK_MODEL_ID:-us.amazon.nova-pro-v1:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

if ! aws ecr describe-repositories --repository-names "$REPOSITORY" --region "$REGION" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$REPOSITORY" --region "$REGION" >/dev/null
fi
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

if ! docker run --privileged --rm tonistiigi/binfmt --install arm64; then
  echo "arm64 binfmt setup failed; falling back to linux/amd64" >&2
  ARCH="X86_64"
  PLATFORM="linux/amd64"
else
  ARCH="ARM64"
  PLATFORM="linux/arm64"
fi

IMAGE="${REGISTRY}/${REPOSITORY}:latest"
if ! docker buildx build --platform "$PLATFORM" --push -t "$IMAGE" "$ROOT"; then
  if [[ "$ARCH" != "X86_64" ]]; then
    echo "arm64 image build failed; falling back to linux/amd64" >&2
    ARCH="X86_64"
    PLATFORM="linux/amd64"
    docker buildx build --platform "$PLATFORM" --push -t "$IMAGE" "$ROOT"
  else
    exit 1
  fi
fi

VPC_ID="$(aws ec2 describe-vpcs --region "$REGION" --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"
SUBNETS="$(aws ec2 describe-subnets --region "$REGION" --filters Name=vpc-id,Values="$VPC_ID" Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text | tr '\t' ',')"
aws cloudformation deploy \
  --region "$REGION" \
  --stack-name "$STACK_NAME" \
  --template-file "$ROOT/infra/cloudformation.yaml" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    VpcId="$VPC_ID" \
    SubnetIds="$SUBNETS" \
    ImageUri="$IMAGE" \
    BedrockModelId="$MODEL_ID" \
    CpuArchitecture="$ARCH" \
    CreateEcrRepository=false
aws ecs update-service --region "$REGION" --cluster "$STACK_NAME" --service "$STACK_NAME-Service" --force-new-deployment >/dev/null || true
aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK_NAME" \
  --query 'Stacks[0].Outputs[?OutputKey==`DashboardURL`].OutputValue' --output text
