# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "app" {
  name              = "/aws/ec2/${var.project_name}"
  retention_in_days = 30

  tags = merge(var.tags, {
    Name = "${var.project_name}-logs"
  })
}

# CloudWatch Alarm for ALB 5xx errors
resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "${var.project_name}-alb-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "HTTPCode_ELB_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  period              = "60"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "This metric monitors ALB 5xx errors"
  alarm_actions       = [] # Add SNS topic ARN here for notifications

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-alb-5xx-alarm"
  })
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", aws_lb.main.arn_suffix],
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", aws_lb.main.arn_suffix],
            ["AWS/ApplicationELB", "HTTPCode_ELB_5XX_Count", "LoadBalancer", aws_lb.main.arn_suffix]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "ALB Metrics"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/EC2", "CPUUtilization", "AutoScalingGroupName", aws_autoscaling_group.app.name],
            ["AWS/EC2", "NetworkIn", "AutoScalingGroupName", aws_autoscaling_group.app.name],
            ["AWS/EC2", "NetworkOut", "AutoScalingGroupName", aws_autoscaling_group.app.name]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "EC2 Metrics"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/S3", "BucketSizeBytes", "BucketName", var.s3_bucket_name, "StorageType", "StandardStorage"],
            ["AWS/S3", "NumberOfObjects", "BucketName", var.s3_bucket_name, "StorageType", "AllStorageTypes"]
          ]
          period = 86400
          stat   = "Average"
          region = var.aws_region
          title  = "S3 Storage Metrics (Daily)"
          yAxis = {
            left = {
              label = "Bytes / Count",
              showUnits = false
            }
          }
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/S3", "GetRequests", "BucketName", var.s3_bucket_name],
            ["AWS/S3", "PutRequests", "BucketName", var.s3_bucket_name],
            ["AWS/S3", "DeleteRequests", "BucketName", var.s3_bucket_name],
            ["AWS/S3", "HeadRequests", "BucketName", var.s3_bucket_name],
            ["AWS/S3", "ListRequests", "BucketName", var.s3_bucket_name]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
          title  = "S3 Request Metrics"
          stacked = false
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/S3", "4xxErrors", "BucketName", var.s3_bucket_name],
            ["AWS/S3", "5xxErrors", "BucketName", var.s3_bucket_name]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
          title  = "S3 Error Rates"
        }
      }
    ]
  })
}

# Optional: S3 CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "s3_4xx_errors" {
  alarm_name          = "${var.project_name}-s3-4xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "4xxErrors"
  namespace           = "AWS/S3"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "This alarm monitors S3 4xx errors"
  alarm_actions       = [] # Add SNS topic ARN for notifications

  dimensions = {
    BucketName = var.s3_bucket_name
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-s3-4xx-alarm"
  })
}

resource "aws_cloudwatch_metric_alarm" "s3_5xx_errors" {
  alarm_name          = "${var.project_name}-s3-5xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "5xxErrors"
  namespace           = "AWS/S3"
  period              = "300"
  statistic           = "Sum"
  threshold           = "1"
  alarm_description   = "This alarm monitors S3 5xx errors"
  alarm_actions       = [] # Add SNS topic ARN for notifications

  dimensions = {
    BucketName = var.s3_bucket_name
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-s3-5xx-alarm"
  })
}