#!/bin/bash

set -e

INPUT_VIDEO_PATH="${1}"

if [ -z "$INPUT_VIDEO_PATH" ]; then
	echo "Usage: $0 <input_video_path>"
	exit 1
fi

eval "$(conda shell.bash hook)"
conda activate nunifiw3

python -m plus.s1_upscale --input_video_path "$INPUT_VIDEO_PATH"
python -m plus.s2_greenscreen --input_video_path "$INPUT_VIDEO_PATH"
