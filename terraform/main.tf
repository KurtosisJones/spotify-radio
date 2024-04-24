provider "aws" {
  region = var.region
}

variable "replace_bucket" {
  type        = bool
  default     = false
}

resource "aws_s3_bucket" "lambda_bucket" {
  bucket = var.bucket_name
  acl    = "private"

  # Conditional creation or replacement
  count = var.replace_bucket ? 1 : 0
}

data "terraform_remote_state" "existing_state" {
  backend = "s3"
  config = {
    bucket = "my-lambda-bucket"
    key    = "path/to/terraform.tfstate"
    region = "us-east-1"
  }
}

resource "null_resource" "remove_existing_bucket" {
  count = data.terraform_remote_state.existing_state.outputs.lambda_bucket_exists ? 1 : 0

  triggers = {
    lambda_bucket_exists = data.terraform_remote_state.existing_state.outputs.lambda_bucket_exists
  }

  provisioner "local-exec" {
    command = "terraform state rm aws_s3_bucket.lambda_bucket"
  }

  depends_on = [
    aws_lambda_function.example_lambda,
    aws_iam_role.lambda_exec_role,
    aws_iam_role_policy.lambda_policy
  ]
}

resource "aws_lambda_function" "example_lambda" {
  count          = var.replace_bucket ? 1 : 0

  function_name  = "example_lambda"
  s3_bucket      = element(aws_s3_bucket.lambda_bucket.*.bucket, 0)
  s3_key         = "${var.lambda_file_path}/${var.lambda_file_name}"
  handler        = "index.handler"
  runtime        = "nodejs14.x"
  role           = aws_iam_role.lambda_exec_role[0].arn
}

resource "aws_iam_role" "lambda_exec_role" {
  count = var.replace_bucket ? 1 : 0

  name = "lambda_exec_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  count = var.replace_bucket ? 1 : 0

  role   = aws_iam_role.lambda_exec_role[0].id
  policy = data.aws_iam_policy_document.lambda_permissions.json
}

data "aws_iam_policy_document" "lambda_permissions" {
  count = var.replace_bucket ? 1 : 0

  statement {
    actions   = ["s3:GetObject"]
    resources = [aws_s3_bucket.lambda_bucket[0].arn]
  }
}
