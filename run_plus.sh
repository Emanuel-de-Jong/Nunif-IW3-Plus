#!/bin/bash

set -e

INPUT_VIDEO_PATH="${1:-./_out/vid.mp4}"

eval "$(conda shell.bash hook)"
conda activate nunifiw3

python -m plus.s1_upscale --input_video_path "$INPUT_VIDEO_PATH"
python -m plus.s2_greenscreen --input_video_path "$INPUT_VIDEO_PATH"
