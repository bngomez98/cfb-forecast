variable "REGISTRY_IMAGE" {
  default = "cfb-forecast"
}

variable "IMAGE_TAG" {
  default = "local"
}

variable "PUSH" {
  default = false
}

variable "PLATFORMS" {
  default = ["linux/amd64"]
}

group "default" {
  targets = ["cfb-forecast"]
}

target "cfb-forecast" {
  context = "."
  dockerfile = "Dockerfile"
  tags = ["${REGISTRY_IMAGE}:${IMAGE_TAG}"]
  platforms = PLATFORMS
  push = PUSH
  cache-from = ["type=gha"]
  cache-to = ["type=gha,mode=max"]
  labels = {
    "org.opencontainers.image.title" = "CFB Forecast"
    "org.opencontainers.image.description" = "Containerized college football forecasting engine"
  }
}
