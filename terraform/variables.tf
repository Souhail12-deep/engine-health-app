# AWS Configuration
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-north-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "engine-health"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "prod"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["eu-north-1a", "eu-north-1b"]
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDRs"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDRs"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.11.0/24"]
}

# EC2 Configuration
variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "instance_count" {
  description = "Number of EC2 instances"
  type        = number
  default     = 2
}

variable "key_name" {
  description = "Key pair name for SSH access"
  type        = string
  default     = ""
}

variable "ami_id" {
  description = "AMI ID for EC2 instances (leave empty to use latest Amazon Linux 2)"
  type        = string
  default     = ""
}

# S3 Configuration
variable "s3_bucket_name" {
  description = "S3 bucket name for models"
  type        = string
}

# Application Configuration
variable "app_port" {
  description = "Application port"
  type        = number
  default     = 5000
}

# Docker Image
variable "docker_image" {
  description = "Docker image URL (ECR or Docker Hub)"
  type        = string
  default     = "engine-health-app:latest"
}

# SSL Certificate
variable "certificate_arn" {
  description = "ARN of SSL certificate for HTTPS (leave empty for HTTP only)"
  type        = string
  default     = ""
}

# Tags
variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "engine-health"
    ManagedBy   = "terraform"
    Environment = "prod"
  }
}