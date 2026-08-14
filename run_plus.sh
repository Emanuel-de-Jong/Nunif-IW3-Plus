#!/bin/bash

set -e

INPUT_VIDEO_PATH="${1}"

if [ -z "$INPUT_VIDEO_PATH" ]; then
	echo "Usage: $0 <input_video_path>"
	exit 1
fi

eval "$(conda shell.bash hook)"
conda activate nunifiw3

# python -m plus.s1_scene_splits --input_video_path "$INPUT_VIDEO_PATH"
# python -m plus.s2_upscale --input_video_path "$INPUT_VIDEO_PATH"
python -m plus.s3_greenscreen --input_video_path "$INPUT_VIDEO_PATH"
python -m plus.s4_fisheye --input_video_path "$INPUT_VIDEO_PATH"
