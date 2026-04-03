#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-flickr}"
device="${2:--1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

python node_classification_inductive.py \
  --device "${device}" \
  --dataset "${dataset}"
