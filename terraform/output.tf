output "lambda_function_name" {
  value = aws_lambda_function.example_lambda.function_name
}

output "s3_bucket_name" {
  value = element(aws_s3_bucket.lambda_bucket[*].bucket, 0)
}
