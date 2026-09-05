#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="${STACK_NAME:-margai}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUCKET="${MARG_S3_BUCKET:-$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK_NAME" --query 'Stacks[0].Outputs[?OutputKey==`Bucket`].OutputValue' --output text)}"

for survey in pothole_cars pothole_kumasi; do
  aws s3 sync "$ROOT/outputs/$survey" "s3://$BUCKET/$survey/" --region "$REGION"
  aws dynamodb put-item --region "$REGION" --table-name margai-surveys \
    --item "{\"survey_id\":{\"S\":\"$survey\"},\"status\":{\"S\":\"done\"}}" >/dev/null
done
echo "Seeded $BUCKET"
