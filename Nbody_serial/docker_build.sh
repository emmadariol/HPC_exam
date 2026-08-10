#!/usr/bin/env bash
set -euo pipefail

image="${IMAGE:-hpc-nbody}"

docker build -t "$image" .
echo "built Docker image: $image"
