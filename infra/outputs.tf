locals {
  # The API returns the address in CIDR form, so the prefix comes off before
  # the value reaches an ssh or rsync command.
  public_ip = try(
    split("/", nebius_compute_v1_instance.trainer.status.network_interfaces[0].public_ip_address.address)[0],
    "PENDING",
  )
}

output "instance_id" {
  description = "Instance ID, for nebius CLI commands."
  value       = nebius_compute_v1_instance.trainer.id
}

output "public_ip" {
  description = "Public address for SSH."
  value       = local.public_ip
}

output "ssh" {
  description = "Ready-made SSH command."
  value       = "ssh ${var.ssh_username}@${local.public_ip}"
}

output "rsync_data" {
  description = "Copies the datasets the training and evaluation steps read."
  value = join(" ", [
    "rsync -av ../data/train.jsonl ../data/valid.jsonl ../data/passages.jsonl",
    "${var.ssh_username}@${local.public_ip}:~/hemingway-sft/data/",
  ])
}
