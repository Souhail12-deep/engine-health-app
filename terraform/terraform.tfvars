# AWS Configuration
aws_region = "eu-north-1"
project_name = "engine-health"
environment = "prod"

# VPC Configuration
vpc_cidr = "10.0.0.0/16"
availability_zones = ["eu-north-1a", "eu-north-1b"]
public_subnet_cidrs = ["10.0.1.0/24", "10.0.2.0/24"]
private_subnet_cidrs = ["10.0.10.0/24", "10.0.11.0/24"]

# EC2 Configuration
instance_type = "t3.micro"
instance_count = 2
key_name = "engine-health-key-v2"  # Your key name

# S3 Configuration - Your existing bucket
s3_bucket_name = "engine-health-models-20260304-112538-28227"

# Application Configuration
app_port = 5000
docker_image = "006250192280.dkr.ecr.eu-north-1.amazonaws.com/engine-health-app:latest"

# SSL Certificate (leave empty for HTTP only)
certificate_arn = ""

# Tags
tags = {
  Project     = "engine-health"
  ManagedBy   = "terraform"
  Environment = "prod"
  Owner       = "souhail"
}