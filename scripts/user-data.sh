#!/bin/bash
set -x
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "========================================="
echo "Starting user-data script at $(date)"
echo "========================================="

# Variables passed from Terraform
DOCKER_IMAGE="${DOCKER_IMAGE}"
S3_BUCKET="${S3_BUCKET}"
ENVIRONMENT="${ENVIRONMENT}"
AWS_REGION="${AWS_REGION}"

echo "DOCKER_IMAGE = $DOCKER_IMAGE"
echo "S3_BUCKET = $S3_BUCKET"
echo "ENVIRONMENT = $ENVIRONMENT"
echo "AWS_REGION = $AWS_REGION"

# Update system
echo "Updating system..."
yum update -y

# Install Docker
echo "Installing Docker..."
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install AWS CLI v2
echo "Installing AWS CLI..."
yum install -y unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
./aws/install
rm -rf aws awscliv2.zip

# Login to ECR
echo "Logging into ECR..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Pull Docker image
echo "Pulling Docker image: $DOCKER_IMAGE"
docker pull $DOCKER_IMAGE

# Create directory for models
echo "Creating models directory..."
mkdir -p /app/models

# Download models from S3
echo "Downloading models from S3..."
aws s3 cp s3://$S3_BUCKET/models/ /app/models/ --recursive || echo "Warning: Model download failed, continuing anyway..."

# List downloaded models for debugging
echo "Models downloaded:"
ls -la /app/models/
ls -la /app/models/anodet_models/ 2>/dev/null || echo "No anodet_models found"
ls -la /app/models/rul_models/ 2>/dev/null || echo "No rul_models found"

# Stop any existing container
docker stop engine-health-app || true
docker rm engine-health-app || true

# Run container with volume mount for models
echo "Starting container with model volume..."
docker run -d \
    --name engine-health-app \
    --restart always \
    -p 5000:5000 \
    -e S3_BUCKET=$S3_BUCKET \
    -e ENVIRONMENT=$ENVIRONMENT \
    -e AWS_REGION=$AWS_REGION \
    -v /app/models:/app/models \
    $DOCKER_IMAGE

# Check if container is running
sleep 5
echo "Container status:"
docker ps -a | grep engine-health-app
echo "Container logs:"
docker logs engine-health-app --tail 30

# Test locally
echo "Testing application locally..."
curl -s http://localhost:5000/health || echo "Health check failed - check logs above"

echo "========================================="
echo "User data script completed at $(date)"
echo "========================================="