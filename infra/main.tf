provider "nebius" {
  parent_id = var.project_id
}

locals {
  ssh_public_key = trimspace(file(pathexpand(var.ssh_public_key_path)))
}

resource "nebius_compute_v1_disk" "boot" {
  parent_id      = var.project_id
  name           = "${var.name}-boot"
  type           = "NETWORK_SSD"
  size_gibibytes = var.boot_disk_gibibytes

  source_image_family = {
    image_family = var.image_family
    parent_id    = var.image_parent_id
  }
}

resource "nebius_compute_v1_instance" "trainer" {
  parent_id = var.project_id
  name      = var.name
  stopped   = var.stopped

  resources = {
    platform = var.platform
    preset   = var.preset
  }

  boot_disk = {
    attach_mode   = "READ_WRITE"
    existing_disk = { id = nebius_compute_v1_disk.boot.id }
  }

  network_interfaces = [{
    name              = "eth0"
    subnet_id         = var.subnet_id
    ip_address        = {}
    public_ip_address = {}
  }]

  cloud_init_user_data = templatefile("${path.module}/cloud-init.yaml.tftpl", {
    ssh_username   = var.ssh_username
    ssh_public_key = local.ssh_public_key
  })
}
