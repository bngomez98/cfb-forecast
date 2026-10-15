variable "REGISTRY_IMAGE" {
  default = "cfb-forecast"
}

variable "IMAGE_TAG" {
  default = "local"
}

variable "PLATFORMS" {
  default = ["linux/amd64"]
}

group "default" {
  targets = ["cfb-forecast", "cfb-dashboard"]
}

target "cfb-dashboard" {
  context = "."
  dockerfile = "dashboard/Dockerfile"
  tags = ["${REGISTRY_IMAGE}-dashboard:${IMAGE_TAG}"]
  platforms = PLATFORMS
  # Each image keeps its own cache. With one shared scope, every build of one
  # image overwrote the other's cache and both rebuilt from scratch.
  cache-from = ["type=gha,scope=cfb-dashboard"]
  cache-to = ["type=gha,scope=cfb-dashboard,mode=max"]
  labels = {
    "org.opencontainers.image.title" = "CFB Forecast Dashboard"
    "org.opencontainers.image.description" = "Streamlit dashboard for weekly CFB predictions"
  }
}

target "cfb-forecast" {
  context = "."
  dockerfile = "Dockerfile"
  tags = ["${REGISTRY_IMAGE}:${IMAGE_TAG}"]
  platforms = PLATFORMS
  cache-from = ["type=gha,scope=cfb-forecast"]
  cache-to = ["type=gha,scope=cfb-forecast,mode=max"]
  labels = {
    "org.opencontainers.image.title" = "CFB Forecast"
    "org.opencontainers.image.description" = "Containerized college football forecasting engine"
  }
}
