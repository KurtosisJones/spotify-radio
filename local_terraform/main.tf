terraform {
}

provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "s3_bucket" {
    bucket = "dataandmodel20240505"
    tags = {
        Name =          "Data and Model Scripts"
        Envrionment = "latest"
    }
}

# resource "aws_s3_bucket_acl" "s3_bucket_acl" {
#     bucket = aws_s3_bucket.s3_bucket.id
#     acl =   "private"
# }

resource "aws_iam_role" "sagemaker_role" {
  name = "sagemaker_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_s3_access" {
  name = "sagemaker_s3_access"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ],
        Effect = "Allow",
        Resource = [
          "${aws_s3_bucket.s3_bucket.arn}",
          "${aws_s3_bucket.s3_bucket.arn}/*"
        ]
      },
    ]
  })
}