variable "region" {
  default = "us-west-2"
}

variable "bucket_name" {
  default = "my-lambda-bucket"
}

variable "lambda_file_path" {
  default = "lambda"
}

variable "lambda_file_name" {
  default = "lambda_function.zip"
}
