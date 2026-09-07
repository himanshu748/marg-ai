#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="${STACK_NAME:-margai}"
REPOSITORY="${ECR_REPOSITORY:-margai}"
MODEL_ID="${MARG_BEDROCK_MODEL_ID:-us.amazon.nova-pro-v1:0}"
AGENT_LLM="${MARG_AGENT_LLM:-mock}"
READ_ONLY="${MARG_DEPLOY_READ_ONLY:-true}"
CERTIFICATE_ARN="${MARG_CERTIFICATE_ARN:-}"
DASHBOARD_DOMAIN="${MARG_DASHBOARD_DOMAIN:-}"
API_TOKEN_SECRET_ARN="${MARG_API_TOKEN_SECRET_ARN:-}"
ARCH="${MARG_CPU_ARCHITECTURE:-ARM64}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$READ_ONLY" in true|false) ;; *) echo "MARG_DEPLOY_READ_ONLY must be true or false" >&2; exit 1 ;; esac
case "$AGENT_LLM" in mock|bedrock) ;; *) echo "MARG_AGENT_LLM must be mock or bedrock" >&2; exit 1 ;; esac
case "$ARCH" in
  ARM64) PLATFORM="linux/arm64" ;;
  X86_64) PLATFORM="linux/amd64" ;;
  *) echo "MARG_CPU_ARCHITECTURE must be ARM64 or X86_64" >&2; exit 1 ;;
esac
if [[ -n "$CERTIFICATE_ARN" && -z "$DASHBOARD_DOMAIN" ]]; then
  echo "Set MARG_DASHBOARD_DOMAIN to a DNS name covered by the ACM certificate" >&2
  exit 1
fi
if [[ -n "$API_TOKEN_SECRET_ARN" && -z "$CERTIFICATE_ARN" ]]; then
  echo "API token authentication requires an HTTPS certificate" >&2
  exit 1
fi
if [[ "$READ_ONLY" == "false" && ( -z "$CERTIFICATE_ARN" || -z "$API_TOKEN_SECRET_ARN" ) ]]; then
  echo "Writable deployment requires MARG_CERTIFICATE_ARN and MARG_API_TOKEN_SECRET_ARN" >&2
  exit 1
fi
if command -v cfn-lint >/dev/null 2>&1; then
  cfn-lint "$ROOT/infra/cloudformation.yaml"
elif [[ -x "$ROOT/.venv/bin/cfn-lint" ]]; then
  "$ROOT/.venv/bin/cfn-lint" "$ROOT/infra/cloudformation.yaml"
else
  echo "Install cfn-lint to validate the template before deployment" >&2
  exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

if ! aws ecr describe-repositories --repository-names "$REPOSITORY" --region "$REGION" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$REPOSITORY" --region "$REGION" >/dev/null
fi
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

IMAGE_TAG="${IMAGE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
IMAGE="${REGISTRY}/${REPOSITORY}:${IMAGE_TAG}"
docker buildx build --platform "$PLATFORM" --push -t "$IMAGE" "$ROOT"

VPC_ID="$(aws ec2 describe-vpcs --region "$REGION" --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"
if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then
  echo "No default VPC is available in $REGION; deploy the template with explicit VpcId and SubnetIds" >&2
  exit 1
fi
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
    AgentLlm="$AGENT_LLM" \
    ReadOnly="$READ_ONLY" \
    CertificateArn="$CERTIFICATE_ARN" \
    DashboardDomain="$DASHBOARD_DOMAIN" \
    ApiTokenSecretArn="$API_TOKEN_SECRET_ARN" \
    CpuArchitecture="$ARCH" \
    CreateEcrRepository=false
aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK_NAME" \
  --query 'Stacks[0].Outputs[?OutputKey==`DashboardURL`].OutputValue' --output text
