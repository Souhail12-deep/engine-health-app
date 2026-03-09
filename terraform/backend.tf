terraform {
  backend "s3" {
    bucket = "engine-health-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "eu-north-1"
  }
}