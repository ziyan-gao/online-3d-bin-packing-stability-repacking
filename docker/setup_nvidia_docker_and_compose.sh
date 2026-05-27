#!/usr/bin/env bash

set -euo pipefail

# Installs NVIDIA Container Toolkit for Docker on Ubuntu/Debian
# and configures Docker to expose GPUs to containers.
# Run this script on the HOST machine (not inside the dev container).
#
# Usage:
#   sudo bash docker/setup_nvidia_docker_and_compose.sh
#   sudo bash docker/setup_nvidia_docker_and_compose.sh --gpu-check

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root (example: sudo bash $0)"
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script currently supports apt-based systems (Ubuntu/Debian)."
  exit 1
fi

GPU_CHECK=false
if [[ "${1:-}" == "--gpu-check" ]]; then
  GPU_CHECK=true
elif [[ "${1:-}" == "--compose-up" ]]; then
  echo "--compose-up is deprecated; use --gpu-check instead."
  GPU_CHECK=true
elif [[ -n "${1:-}" ]]; then
  echo "Unknown argument: $1"
  echo "Usage: sudo bash $0 [--gpu-check]"
  exit 1
fi

echo "[1/6] Installing required packages..."
apt-get update
apt-get install -y --no-install-recommends curl gnupg ca-certificates

echo "[2/6] Adding NVIDIA Container Toolkit keyring..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

echo "[3/6] Adding NVIDIA Container Toolkit apt repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "[4/6] Installing nvidia-container-toolkit..."
apt-get update
apt-get install -y nvidia-container-toolkit nvidia-modprobe

echo "[5/6] Configuring Docker runtime for NVIDIA..."
nvidia-ctk runtime configure --runtime=docker

echo "[6/6] Restarting Docker..."
systemctl restart docker

if command -v nvidia-modprobe >/dev/null 2>&1; then
  nvidia-modprobe -u -c=0 || true
fi

echo
echo "Install/config complete."
echo "Run this test command:"
echo "docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"

if [[ "${GPU_CHECK}" == "true" ]]; then
  echo
  echo "Requested: docker compose GPU check"

  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker CLI not found; skipping GPU check."
    exit 0
  fi

  if [[ ! -f "${COMPOSE_FILE}" ]]; then
    echo "No docker-compose.yml found at ${COMPOSE_FILE}; skipping GPU check."
    exit 0
  fi

  if [[ -n "${SUDO_USER:-}" ]]; then
    sudo -u "${SUDO_USER}" docker compose -f "${COMPOSE_FILE}" --project-directory "${REPO_ROOT}" --profile gpu run --rm train-gpu \
      python -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
  else
    docker compose -f "${COMPOSE_FILE}" --project-directory "${REPO_ROOT}" --profile gpu run --rm train-gpu \
      python -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
  fi
fi
