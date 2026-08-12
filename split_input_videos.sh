#!/bin/bash

set -e

VIDEO_DIR_PATH="${1:-./_out}"

eval "$(conda shell.bash hook)"
conda activate nunifiw3

python -u -m plus.split_input_videos --path "$VIDEO_DIR_PATH"
