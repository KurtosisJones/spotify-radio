provider "aws" {
  region = var.region
}

resource "aws_s3_bucket" "lambda_bucket" {
  bucket = var.bucket_name
  acl    = "private"
}

resource "aws_lambda_function" "example_lambda" {
  function_name = "example_lambda"
  
  s3_bucket = aws_s3_bucket.lambda_bucket.bucket
  s3_key    = "${var.lambda_file_path}/${var.lambda_file_name}"

  handler = "index.handler"
  runtime = "nodejs14.x"

  role = aws_iam_role.lambda_exec_role.arn
}

resource "aws_iam_role" "lambda_exec_role" {
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
  role   = aws_iam_role.lambda_exec_role.id
  policy = data.aws_iam_policy_document.lambda_permissions.json
}

data "aws_iam_policy_document" "lambda_permissions" {
  statement {
    actions   = ["s3:GetObject"]
    resources = [aws_s3_bucket.lambda_bucket.arn]
  }
}
