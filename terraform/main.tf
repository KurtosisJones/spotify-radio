provider "aws" {
  region = "us-east-1"
}

terraform {
  backend "s3" {
    bucket         = "terraform-state-bucket-radio"
    key            = "state/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "lock-table"
    encrypt        = true
  }
}

resource "aws_s3_bucket" "my_bucket" {
  bucket = "spotify-radio-latest"
}

# data "aws_iam_policy_document" "assume_role" {
#   statement {
#     effect = "Allow"

#     principals {
#       type        = "Service"
#       identifiers = ["lambda.amazonaws.com"]
#     }

#     actions = ["sts:AssumeRole"]
#   }
# }

# data "aws_iam_policy_document" "s3_access" {
#   statement {
#     effect = "Allow"

#     actions = [
#       "s3:PutObject",
#       "s3:GetObject",
#       "s3:DeleteObject"
#     ]

#     resources = [
#       "${aws_s3_bucket.my_bucket.arn}/*"
#     ]
#   }
# }

# resource "aws_iam_policy" "s3_policy" {
#   name   = "s3_policy"
#   policy = data.aws_iam_policy_document.s3_access.json
# }

# resource "aws_iam_role" "iam_for_lambda" {
#   name               = "iam_for_lambda"
#   assume_role_policy = data.aws_iam_policy_document.assume_role.json
# }

# resource "aws_iam_role_policy_attachment" "s3_policy_attach" {
#   role       = aws_iam_role.iam_for_lambda.name
#   policy_arn = aws_iam_policy.s3_policy.arn
# }

# data "archive_file" "lambda" {
#   type = "zip"
#   source_file = "../lambda/print_hello.py"
#   output_path = "lambda_function_payload.zip"
# }

# resource "aws_lambda_function" "lambda_function" {
#   filename         = data.archive_file.lambda.output_path
#   function_name    = "lambda_function_name"
#   role             = aws_iam_role.iam_for_lambda.arn
#   handler          = "print_hello.handler"

#   source_code_hash = data.archive_file.lambda.output_base64sha256
#   runtime          = "python3.8"

#   environment {
#     variables = {
#       foo = "bar"
#     }
#   }
# }
