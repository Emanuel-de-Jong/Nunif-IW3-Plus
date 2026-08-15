#!/bin/bash
set -e

INPUT_VIDEO_PATH="$1"
if [ -z "$INPUT_VIDEO_PATH" ]; then
	echo "Usage: $0 <input_video_path>"
	exit 1
elif [ ! -f "$INPUT_VIDEO_PATH" ]; then
	echo "Input video not found: $INPUT_VIDEO_PATH"
	exit 1
fi

INPUT_VIDEO_DIR="$(cd "$(dirname "$INPUT_VIDEO_PATH")" && pwd)"
INPUT_FILENAME="$(basename "$INPUT_VIDEO_PATH")"
INPUT_FILENAME="${INPUT_FILENAME%.*}"
PLUS_DIR="$INPUT_VIDEO_DIR/plus"
TMP_DIR="$PLUS_DIR/tmp"
RESULT_PATH="$PLUS_DIR/${INPUT_FILENAME}_result.mp4"
if [ -f "$RESULT_PATH" ]; then
	echo "Result already exists: $RESULT_PATH"
	exit 0
fi

eval "$(conda shell.bash hook)"
conda deactivate
conda activate nunifiw3
mkdir -p "$TMP_DIR"

get_name() {
	local name="${1##*/}"
	name="${name%.*}"
	name="${name%_1_scene}"
	name="${name%_2_upscale}"
	name="${name%_3_green}"
	printf %s "$name"
}

get_step_output_path() {
	printf '%s/%s_%s.mp4\n' "$TMP_DIR" "$(get_name "$1")" "$2"
}

get_result_path() {
	printf '%s/%s_result.mp4\n' "$PLUS_DIR" "$(get_name "$1")"
}

advance_inputs() {
	local i
	for i in "${!pipeline_inputs[@]}"; do
		pipeline_inputs[i]="$(get_step_output_path "${pipeline_inputs[i]}" "$1")"
	done
}

all_results_exist() {
	local input_path
	for input_path in "${pipeline_inputs[@]}"; do
		[ -f "$(get_result_path "$input_path")" ] || return 1
	done
	return 0
}

run_scene_splits() {
	printf "=== STEP 1: SCENE SPLITS ===\n"
	python -m plus.s1_scene_splits --input_video_path "$INPUT_VIDEO_PATH" --output_dir "$TMP_DIR"
	shopt -s nullglob
	pipeline_inputs=("$TMP_DIR/${INPUT_FILENAME}_"*_1_scene.mp4)
	shopt -u nullglob
	if [ "${#pipeline_inputs[@]}" -eq 0 ]; then
		echo "No scene split videos found in: $TMP_DIR"
		exit 1
	fi
	if all_results_exist; then
		echo "All results already exist in: $PLUS_DIR"
		exit 0
	fi
}

run_upscale() {
	printf "\n\n=== STEP 2: UPSCALE ===\n"
	local input_path output_path
	for input_path in "${pipeline_inputs[@]}"; do
		output_path="$(get_step_output_path "$input_path" 2_upscale)"
		if [ -f "$output_path" ]; then
			echo "Upscale output already exists: $output_path"
			continue
		fi
		python -m plus.s2_upscale --input_video_path "$input_path" --output_video_path "$output_path"
	done
	advance_inputs 2_upscale
}

run_greenscreen() {
	printf "\n\n=== STEP 3: GREENSCREEN ===\n"
	local input_path output_path
	conda deactivate
	conda activate sam3
	for input_path in "${pipeline_inputs[@]}"; do
		output_path="$(get_step_output_path "$input_path" 3_green)"
		if [ -f "$output_path" ]; then
			echo "Greenscreen output already exists: $output_path"
			continue
		fi
		python -m plus.s3_greenscreen --input_video_path "$input_path" --output_dir "$TMP_DIR" --output_video_path "$output_path"
	done
	conda deactivate
	conda activate nunifiw3
	advance_inputs 3_green
}

run_fisheye() {
	printf "\n\n=== STEP 4: FISHEYE ===\n"
	local input_path output_path result_path
	for input_path in "${pipeline_inputs[@]}"; do
		output_path="$(get_step_output_path "$input_path" 4_fish)"
		result_path="$(get_result_path "$input_path")"
		if [ -f "$result_path" ]; then
			echo "Result already exists: $result_path"
			continue
		fi
		if [ -f "$output_path" ]; then
			echo "Fisheye output already exists: $output_path"
		else
			python -m plus.s4_fisheye --input_video_path "$input_path" --fisheye_input_video_path "$input_path" --output_video_path "$output_path"
		fi
		cp "$output_path" "$result_path"
	done
}

pipeline_inputs=("$INPUT_VIDEO_PATH")

# run_scene_splits
# run_upscale
run_greenscreen
run_fisheye

printf "\n=== DONE! ===\n"
