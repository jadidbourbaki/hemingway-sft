variable "nebius_profile" {
  description = "Name of the nebius CLI profile whose credentials the provider uses. Run `nebius profile list`."
  type        = string
}

variable "project_id" {
  description = "Nebius project that owns the instance and its disk. Run `nebius config list`."
  type        = string
}

variable "subnet_id" {
  description = "Subnet the instance attaches to. Run `nebius vpc subnet list` and take the default one."
  type        = string
}

variable "name" {
  description = "Name given to the instance and used as a prefix for its disk."
  type        = string
  default     = "hemingway-trainer"
}

variable "platform" {
  description = "GPU platform. An L40S holds the run in 48GB, so an H100 buys nothing here."
  type        = string
  default     = "gpu-l40s-a"
}

variable "preset" {
  description = "Platform preset. Host memory matters because loading an 8B checkpoint spikes it."
  type        = string
  default     = "1gpu-16vcpu-64gb"
}

variable "image_family" {
  description = "Boot image family. The CUDA builds ship the NVIDIA driver already."
  type        = string
  default     = "mk8s-worker-node-v-1-34-ubuntu24.04-cuda13.0"
}

variable "image_parent_id" {
  description = "Project holding the public images."
  type        = string
  default     = "project-e00public-images"
}

variable "boot_disk_gibibytes" {
  description = "Boot disk size. Model weights, the Hugging Face cache, and a torch venv need room."
  type        = number
  default     = 200
}

variable "ssh_public_key_path" {
  description = "Path to the public key authorised for SSH. Reading the file beats pasting the key, because a missing file fails the plan instead of building an unreachable instance."
  type        = string
}

variable "ssh_username" {
  description = "Login created on the instance."
  type        = string
  default     = "trainer"
}

variable "stopped" {
  description = "Stop the instance without destroying it, which keeps the disk and its model cache."
  type        = bool
  default     = false
}
