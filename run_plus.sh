#!/bin/bash

set -e

INPUT_VIDEO_PATH="${1}"

if [ -z "$INPUT_VIDEO_PATH" ]; then
	echo "Usage: $0 <input_video_path>"
	exit 1
fi

if [ ! -f "$INPUT_VIDEO_PATH" ]; then
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
conda activate nunifiw3

mkdir -p "$TMP_DIR"

get_step_output_path() {
	local input_path="$1"
	local step_suffix="$2"
	local output_name
	output_name="$(basename "$input_path")"
	output_name="${output_name%.*}"
	output_name="${output_name%_1_scene}"
	output_name="${output_name%_2_upscale}"
	output_name="${output_name%_3_green}"
	printf '%s/%s_%s.mp4\n' "$TMP_DIR" "$output_name" "$step_suffix"
}

get_result_path() {
	local input_path="$1"
	local result_name
	result_name="$(basename "$input_path")"
	result_name="${result_name%.*}"
	result_name="${result_name%_1_scene}"
	result_name="${result_name%_2_upscale}"
	result_name="${result_name%_3_green}"
	printf '%s/%s_result.mp4\n' "$PLUS_DIR" "$result_name"
}

scene_splits_ran=false
upscale_ran=false
greenscreen_ran=false

run_scene_splits() {
	printf "=== STEP 1: SCENE SPLITS ===\n"
	scene_splits_ran=true
	python -m plus.s1_scene_splits --input_video_path "$INPUT_VIDEO_PATH" --output_dir "$TMP_DIR"
}

run_upscale() {
	printf "\n\n=== STEP 2: UPSCALE ===\n"
	local input_path
	local output_path
	upscale_ran=true
	for input_path in "${pipeline_inputs[@]}"; do
		output_path="$(get_step_output_path "$input_path" "2_upscale")"
		if [ -f "$output_path" ]; then
			echo "Upscale output already exists: $output_path"
			continue
		fi
		python -m plus.s2_upscale --input_video_path "$input_path" --output_video_path "$output_path"
	done
}

run_greenscreen() {
	printf "\n\n=== STEP 3: GREENSCREEN ===\n"
	local input_path
	local output_path
	greenscreen_ran=true
	for input_path in "${pipeline_inputs[@]}"; do
		output_path="$(get_step_output_path "$input_path" "3_green")"
		if [ -f "$output_path" ]; then
			echo "Greenscreen output already exists: $output_path"
			continue
		fi
		python -m plus.s3_greenscreen --input_video_path "$input_path" --output_dir "$TMP_DIR" --output_video_path "$output_path"
	done
}

run_fisheye() {
	printf "\n\n=== STEP 4: FISHEYE ===\n"
	local input_path
	local output_path
	local result_path
	for input_path in "${pipeline_inputs[@]}"; do
		output_path="$(get_step_output_path "$input_path" "4_fish")"
		result_path="$(get_result_path "$input_path")"
		if [ -f "$result_path" ]; then
			echo "Result already exists: $result_path"
			continue
		fi
		if [ ! -f "$output_path" ]; then
			python -m plus.s4_fisheye --input_video_path "$input_path" --fisheye_input_video_path "$input_path" --output_video_path "$output_path"
		else
			echo "Fisheye output already exists: $output_path"
		fi
		cp "$output_path" "$result_path"
	done
}

run_scene_splits

if [ "$scene_splits_ran" = true ]; then
	shopt -s nullglob
	pipeline_inputs=("$TMP_DIR/${INPUT_FILENAME}_"*_1_scene.mp4)
	shopt -u nullglob
	if [ "${#pipeline_inputs[@]}" -eq 0 ]; then
		echo "No scene split videos found in: $TMP_DIR"
		exit 1
	fi
else
	pipeline_inputs=("$INPUT_VIDEO_PATH")
fi

all_results_exist=true
for input_path in "${pipeline_inputs[@]}"; do
	if [ ! -f "$(get_result_path "$input_path")" ]; then
		all_results_exist=false
		break
	fi
done

if [ "$all_results_exist" = true ]; then
	echo "All results already exist in: $PLUS_DIR"
	exit 0
fi

# run_upscale

if [ "$upscale_ran" = true ]; then
	for input_index in "${!pipeline_inputs[@]}"; do
		pipeline_inputs[$input_index]="$(get_step_output_path "${pipeline_inputs[$input_index]}" "2_upscale")"
	done
fi

run_greenscreen

if [ "$greenscreen_ran" = true ]; then
	for input_index in "${!pipeline_inputs[@]}"; do
		pipeline_inputs[$input_index]="$(get_step_output_path "${pipeline_inputs[$input_index]}" "3_green")"
	done
fi

run_fisheye

printf "\n=== DONE! ===\n"
