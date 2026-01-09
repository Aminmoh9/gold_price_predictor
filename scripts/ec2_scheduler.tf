# Terraform configuration for EC2 auto start/stop
# This uses AWS Lambda + EventBridge to schedule EC2 start/stop
# Saves money by only running EC2 when needed

# Variables
variable "ec2_instance_id" {
  description = "The EC2 instance ID to manage"
  type        = string
}

variable "start_cron" {
  description = "Cron expression for starting EC2 (UTC)"
  default     = "cron(0 8 * * ? *)"  # 8 AM UTC daily
}

variable "stop_cron" {
  description = "Cron expression for stopping EC2 (UTC)"
  default     = "cron(0 22 * * ? *)"  # 10 PM UTC daily
}

# IAM Role for Lambda
resource "aws_iam_role" "ec2_scheduler_role" {
  name = "ec2_scheduler_lambda_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "ec2_scheduler_policy" {
  name = "ec2_scheduler_policy"
  role = aws_iam_role.ec2_scheduler_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:StartInstances",
          "ec2:StopInstances",
          "ec2:DescribeInstances"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

# Lambda function to start EC2
resource "aws_lambda_function" "start_ec2" {
  filename         = "lambda_ec2_scheduler.zip"
  function_name    = "start_ec2_instance"
  role            = aws_iam_role.ec2_scheduler_role.arn
  handler         = "index.start_handler"
  runtime         = "python3.9"
  timeout         = 30

  environment {
    variables = {
      INSTANCE_ID = var.ec2_instance_id
    }
  }
}

# Lambda function to stop EC2
resource "aws_lambda_function" "stop_ec2" {
  filename         = "lambda_ec2_scheduler.zip"
  function_name    = "stop_ec2_instance"
  role            = aws_iam_role.ec2_scheduler_role.arn
  handler         = "index.stop_handler"
  runtime         = "python3.9"
  timeout         = 30

  environment {
    variables = {
      INSTANCE_ID = var.ec2_instance_id
    }
  }
}

# EventBridge rules for scheduling
resource "aws_cloudwatch_event_rule" "start_ec2_rule" {
  name                = "start_ec2_schedule"
  description         = "Start EC2 instance on schedule"
  schedule_expression = var.start_cron
}

resource "aws_cloudwatch_event_rule" "stop_ec2_rule" {
  name                = "stop_ec2_schedule"
  description         = "Stop EC2 instance on schedule"
  schedule_expression = var.stop_cron
}

# EventBridge targets
resource "aws_cloudwatch_event_target" "start_target" {
  rule      = aws_cloudwatch_event_rule.start_ec2_rule.name
  target_id = "StartEC2"
  arn       = aws_lambda_function.start_ec2.arn
}

resource "aws_cloudwatch_event_target" "stop_target" {
  rule      = aws_cloudwatch_event_rule.stop_ec2_rule.name
  target_id = "StopEC2"
  arn       = aws_lambda_function.stop_ec2.arn
}

# Lambda permissions for EventBridge
resource "aws_lambda_permission" "allow_start_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.start_ec2.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.start_ec2_rule.arn
}

resource "aws_lambda_permission" "allow_stop_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.stop_ec2.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.stop_ec2_rule.arn
}
