#!/bin/bash
set -e

ENVIRONMENT=${1:-dev}
echo "🚀 Deploying to $ENVIRONMENT..."

cd terraform
terraform init
terraform apply -auto-approve -var="environment=$ENVIRONMENT"

ALB_DNS=$(terraform output -raw alb_dns_name)
echo "✅ Application deployed at: http://$ALB_DNS"