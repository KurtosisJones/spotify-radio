provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "my_bucket" {
  bucket = "spotify-radio-latest"
}

data "aws_iam_policy_document" "assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "iam_for_lambda" {
  name               = "iam_for_lambda"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

data "archive_file" "lambda" {
  type = "zip"
  source_file = "lambda/print_hello.py"
  output_path = "lambda_function_payload.zip"
}

resource "aws_lambda_function" "lambda_function" {
  filename         = data.archive_file.lambda.output_path
  function_name    = "lambda_function_name"
  role             = aws_iam_role.iam_for_lambda.arn
  handler          = "print_hello.handler"

  source_code_hash = data.archive_file.lambda.output_base64sha256
  runtime          = "python3.8"

  environment {
    variables = {
      foo = "bar"
    }
  }
}
