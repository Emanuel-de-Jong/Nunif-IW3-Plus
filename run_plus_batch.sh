#!/bin/bash

set -e

INPUT_DIR="${1}"

if [ -z "$INPUT_DIR" ]; then
	echo "Usage: $0 <input_dir>"
	exit 1
fi

eval "$(conda shell.bash hook)"
conda activate nunifiw3

shopt -s nullglob

videos=()

for video in "$INPUT_DIR"/*.{mp4,mov,avi,mkv,webm}; do
	[ -e "$video" ] || continue

	filename="$(basename "$video")"
	basename="${filename%.*}"
	result="$INPUT_DIR/plus/${basename}_matte/${basename}_matte.json"

	if [ -e "$result" ]; then
		continue
	fi

	videos+=("$video")
done

TOTAL_COUNT=${#videos[@]}
CURRENT_COUNT=0

for video in "${videos[@]}"; do
	((CURRENT_COUNT++))

	printf "=== %s (%d/%d) ===\n" "$video" "$CURRENT_COUNT" "$TOTAL_COUNT"

	python -m plus.s1_scene_splits --input_video_path "$video"
	python -m plus.s2_upscale --input_video_path "$video"
	python -m plus.s3_greenscreen --input_video_path "$video"

	printf "\n\n-----------------------------------\n\n"
done
