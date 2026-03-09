#!/bin/bash
set -x
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "========================================="
echo "Starting user-data script at $(date)"
echo "========================================="

DOCKER_IMAGE="${DOCKER_IMAGE}"
S3_BUCKET="${S3_BUCKET}"
ENVIRONMENT="${ENVIRONMENT}"
AWS_REGION="${AWS_REGION}"

# Update system
yum update -y

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install AWS CLI
yum install -y unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
./aws/install
rm -rf aws awscliv2.zip

# Login to ECR
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Pull image
docker pull $DOCKER_IMAGE

# Run container
docker run -d \
    --name engine-health-app \
    --restart always \
    -p 5000:5000 \
    -e S3_BUCKET=$S3_BUCKET \
    -e ENVIRONMENT=$ENVIRONMENT \
    -e AWS_REGION=$AWS_REGION \
    $DOCKER_IMAGE

sleep 5
echo "Container status:"
docker ps -a | grep engine-health-app
echo "Container logs:"
docker logs engine-health-app --tail 30

echo "========================================="
echo "User data script completed at $(date)"
echo "========================================="