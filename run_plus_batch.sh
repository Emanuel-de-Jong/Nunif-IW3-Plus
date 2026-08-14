#!/bin/bash

set -e

INPUT_DIR="${1}"

if [ -z "$INPUT_DIR" ]; then
	echo "Usage: $0 <input_dir>"
	exit 1
fi

shopt -s nullglob

videos=()

for video in "$INPUT_DIR"/*.{mp4,mov,avi,mkv,webm}; do
	[ -e "$video" ] || continue

	filename="$(basename "$video")"
	basename="${filename%.*}"
	result="$INPUT_DIR/plus/${basename}_result.mp4"

	if [ -e "$result" ]; then
		continue
	fi

	videos+=("$video")
done

TOTAL_COUNT=${#videos[@]}
if (( TOTAL_COUNT == 0 )); then
	echo "No unprocessed videos found in: $INPUT_DIR"
	exit 0
fi

CURRENT_COUNT=0

for video in "${videos[@]}"; do
	((++CURRENT_COUNT))

	printf "=== %s (%d/%d) ===\n" "$video" "$CURRENT_COUNT" "$TOTAL_COUNT"

	./run_plus.sh "$video"

	printf "\n\n-----------------------------------\n\n"
done
