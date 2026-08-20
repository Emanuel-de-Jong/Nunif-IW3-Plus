import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import re
import ast
import json
import shutil
import tempfile
import cv2
import torch
import numpy as np
import imageio_ffmpeg
import plus.global_params as g
from fire import Fire
from iw3.zoedepth_model import ZoeDepthModel
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    Sam3VideoModel,
    Sam3VideoProcessor,
)

SAM_HOTSTART_FRAMES = 15


def main(
    input_video_path: str,
    output_dir: str = None,
    output_video_path: str = None,
    vlm_model_id: str = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
    num_sampled_frames: int = 7,
    num_vote_runs: int = 2,
    vlm_temperature: float = 0.2,
    vlm_max_new_tokens: int = 768,
    vlm_max_long_side: int = 1024,
    sam_model_id: str = "facebook/sam3",
    sam_prompt_groups: int = 4,
    sam_max_long_side: int = 720,
    sam_chunk_seconds: float = 1.0,
    sam_chunk_overlap_seconds: float = 0.35,
    sam_video_crf: int = 12,
    sam_video_preset: str = "veryfast",
    sam_compile: bool = False,
    output_alpha_video: bool = True,
    output_instance_videos: bool = False,
    sam_mask_close_kernel: int = 9,
    sam_mask_border_shift: int = 1,
    sam_mask_overlap_gap_fill: int = 40,
    qc_frame_interval: int = 15,
    qc_area_jump_threshold: float = 0.40,
    greenscreen_crf: int = 18,
    greenscreen_preset: str = "medium",
    depth_foreground_threshold: float = 0.2,
    depth_mask_border_shift: int = -10,
    prompt_override: str = None,
    overwrite: bool = False,
):
    device = "cuda"

    video_dir = os.path.dirname(os.path.abspath(input_video_path))
    video_stem = os.path.splitext(os.path.basename(input_video_path))[0]
    if output_dir is None:
        output_dir = os.path.join(video_dir, "plus", "tmp")
    if output_video_path is None:
        output_video_path = os.path.join(output_dir, f"{video_stem}_3_green.mp4")

    output_stem = os.path.splitext(os.path.basename(output_video_path))[0]
    sidecar_path = os.path.join(output_dir, f"{output_stem}.json")
    qc_log_path = os.path.join(output_dir, f"{output_stem}_qc.log")
    greenscreen_path = output_video_path

    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")
    try:
        fps, width, height, frame_count = get_video_properties(video)
    finally:
        video.release()
    if width % 2 != 0:
        raise ValueError(f"SBS video width must be even, got: {width}")

    eye_width = width // 2
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, f".{output_stem}_work")
    state_path = os.path.join(work_dir, "state.json")
    if not os.path.isfile(state_path) and g.should_skip_output(
        greenscreen_path, overwrite
    ):
        return
    os.makedirs(work_dir, exist_ok=True)

    retained_eye_path = os.path.join(output_dir, f"{output_stem}_L.mp4")
    retained_green_path = os.path.join(output_dir, f"{output_stem}_green_L.mp4")
    retained_state_path = os.path.join(output_dir, f"{output_stem}_state.json")
    retained_depth_path = os.path.join(output_dir, f"{output_stem}_depth_L.mp4")
    retained_depth_raw_path = os.path.join(output_dir, f"{output_stem}_depth_L_raw.mp4")
    retained_sam_path = os.path.join(output_dir, f"{output_stem}_sam_L.mp4")
    retained_sam_raw_path = os.path.join(output_dir, f"{output_stem}_sam_L_raw.mp4")

    config = {
        "vlm_model_id": vlm_model_id,
        "num_sampled_frames": num_sampled_frames,
        "num_vote_runs": num_vote_runs,
        "vlm_temperature": vlm_temperature,
        "vlm_max_new_tokens": vlm_max_new_tokens,
        "vlm_max_long_side": vlm_max_long_side,
        "sam_model_id": sam_model_id,
        "sam_prompt_groups": sam_prompt_groups,
        "sam_max_long_side": sam_max_long_side,
        "sam_chunk_seconds": sam_chunk_seconds,
        "sam_chunk_overlap_seconds": sam_chunk_overlap_seconds,
        "sam_video_crf": sam_video_crf,
        "sam_video_preset": sam_video_preset,
        "sam_compile": sam_compile,
        "output_alpha_video": output_alpha_video,
        "output_instance_videos": output_instance_videos,
        "sam_mask_close_kernel": sam_mask_close_kernel,
        "sam_mask_border_shift": sam_mask_border_shift,
        "sam_mask_overlap_gap_fill": sam_mask_overlap_gap_fill,
        "qc_frame_interval": qc_frame_interval,
        "qc_area_jump_threshold": qc_area_jump_threshold,
        "greenscreen_crf": greenscreen_crf,
        "greenscreen_preset": greenscreen_preset,
        "depth_foreground_threshold": depth_foreground_threshold,
        "depth_mask_border_shift": depth_mask_border_shift,
        "prompt_override": prompt_override,
        "device": device,
    }
    state = load_resume_state(state_path, input_video_path, output_stem, config)
    sidecar = {
        "input_video_path": input_video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "config": config,
        "sampled_frame_indices": [],
        "include": [],
        "specific_include": [],
        "exclude": [],
        "sam_prompts": [],
        "eyes": {},
        "outputs": {},
        "failures": [],
    }
    restore_sidecar_from_state(sidecar, state)
    pipeline_complete = False

    try:
        sampled_frame_indices = get_sampled_frame_indices(
            frame_count, num_sampled_frames
        )
        sidecar["sampled_frame_indices"] = [
            int(index) for index in sampled_frame_indices
        ]
        save_resume_state(state_path, state, input_video_path, output_stem, config)
        frames_dir = os.path.join(work_dir, "samples")
        image_paths = get_completed_sample_frames(
            state, frames_dir, len(sampled_frame_indices)
        )
        if image_paths is None:
            image_paths = extract_sample_frames(
                input_video_path,
                sampled_frame_indices,
                eye_width,
                frames_dir,
                vlm_max_long_side,
            )
            if len(image_paths) > 0:
                state["image_paths"] = image_paths
                mark_completed(state, "sample_frames")
                save_resume_state(
                    state_path, state, input_video_path, output_stem, config
                )
        if len(image_paths) == 0:
            sidecar["failures"].append("frame sampling produced no frames")
            print("==> frame sampling produced no frames", flush=True)
            save_sidecar(sidecar_path, sidecar)
            write_qc_log(qc_log_path, sidecar)
            return

        print(f"==> sampled {len(image_paths)} frames for VLM", flush=True)
        if prompt_override is not None:
            prompts = parse_prompt_override(prompt_override)
            include_concepts = []
            specific_include_concepts = prompts
            exclude_concepts = []
            sidecar["prompt_source"] = "override"
            state["prompt_source"] = "override"
            state["include"] = include_concepts
            state["specific_include"] = specific_include_concepts
            state["exclude"] = exclude_concepts
            state["sam_prompts"] = prompts
            mark_completed(state, "prompts")
            save_resume_state(state_path, state, input_video_path, output_stem, config)
        elif is_completed(state, "prompts") and "sam_prompts" in state:
            include_concepts = state.get("include", [])
            specific_include_concepts = state.get("specific_include", [])
            exclude_concepts = state.get("exclude", [])
            prompts = state["sam_prompts"]
            sidecar["prompt_source"] = state.get("prompt_source", "qwen")
        else:
            qwen_prompts = load_qwen_prompts()
            include_concepts, specific_include_concepts, exclude_concepts = (
                identify_foreground(
                    vlm_model_id,
                    device,
                    image_paths,
                    num_vote_runs,
                    vlm_temperature,
                    vlm_max_new_tokens,
                    sidecar,
                    qwen_prompts,
                )
            )
            prompts = build_sam_prompts(
                include_concepts,
                specific_include_concepts,
                sam_prompt_groups,
                qwen_prompts,
            )
            sidecar["prompt_source"] = "qwen"
            state["prompt_source"] = "qwen"
            state["include"] = include_concepts
            state["specific_include"] = specific_include_concepts
            state["exclude"] = exclude_concepts
            state["sam_prompts"] = prompts
            mark_completed(state, "prompts")
            save_resume_state(state_path, state, input_video_path, output_stem, config)

        sidecar["include"] = include_concepts
        sidecar["specific_include"] = specific_include_concepts
        sidecar["exclude"] = exclude_concepts
        sidecar["sam_prompts"] = prompts
        print(f"==> voted include concepts: {include_concepts}", flush=True)
        print(
            f"==> voted specific include concepts: {specific_include_concepts}",
            flush=True,
        )
        print(f"==> voted exclude concepts: {exclude_concepts}", flush=True)
        print(f"==> SAM prompts: {prompts}", flush=True)

        if len(prompts) == 0:
            sidecar["failures"].append("no foreground objects detected")
            print("==> no foreground objects detected", flush=True)
            save_sidecar(sidecar_path, sidecar)
            write_qc_log(qc_log_path, sidecar)
            return

        eye_video_paths = {
            "L": os.path.join(work_dir, "eye_L.mp4"),
            "R": os.path.join(work_dir, "eye_R.mp4"),
        }
        eye_green_paths = {
            "L": os.path.join(work_dir, "green_L.mp4"),
            "R": os.path.join(work_dir, "green_R.mp4"),
        }
        eye_alpha_paths = {
            "L": os.path.join(work_dir, "alpha_L.mp4"),
            "R": os.path.join(work_dir, "alpha_R.mp4"),
        }
        eye_depth_mask_paths = {
            "L": os.path.join(work_dir, "depth_mask_L.mkv"),
            "R": os.path.join(work_dir, "depth_mask_R.mkv"),
        }
        eye_depth_green_paths = {
            "L": os.path.join(work_dir, "depth_L.mp4"),
            "R": None,
        }
        eye_depth_raw_green_paths = {
            "L": os.path.join(work_dir, "depth_L_raw.mp4"),
            "R": None,
        }
        eye_sam_green_paths = {
            "L": os.path.join(work_dir, "sam_L.mp4"),
            "R": None,
        }
        eye_sam_raw_green_paths = {
            "L": os.path.join(work_dir, "sam_L_raw.mp4"),
            "R": None,
        }

        try:
            for eye in ("L", "R"):
                crop_step = f"crop_{eye}"
                if not is_completed_file(state, crop_step, eye_video_paths[eye]):
                    temp_eye_video_path = get_temp_path(eye_video_paths[eye])
                    remove_if_exists(temp_eye_video_path)
                    crop_eye_video(
                        input_video_path,
                        eye,
                        temp_eye_video_path,
                        sam_max_long_side,
                        sam_video_crf,
                        sam_video_preset,
                    )
                    os.replace(temp_eye_video_path, eye_video_paths[eye])
                    mark_completed(state, crop_step)
                    save_resume_state(
                        state_path, state, input_video_path, output_stem, config
                    )

            depth_mask_paths = {}
            if depth_foreground_threshold > 0:
                pending_depth_eyes = []
                for eye in ("L", "R"):
                    depth_step = f"depth_{eye}"
                    if not is_completed_file(
                        state, depth_step, eye_depth_mask_paths[eye]
                    ):
                        pending_depth_eyes.append(eye)
                    depth_mask_paths[eye] = eye_depth_mask_paths[eye]
                if len(pending_depth_eyes) > 0:
                    depth_model = create_depth_model()
                    try:
                        for eye in pending_depth_eyes:
                            print(
                                f"==> computing depth masks for eye {eye}", flush=True
                            )
                            depth_step = f"depth_{eye}"
                            temp_depth_mask_path = get_temp_path(
                                eye_depth_mask_paths[eye]
                            )
                            remove_if_exists(temp_depth_mask_path)
                            compute_eye_depth_mask_video(
                                depth_model,
                                eye_video_paths[eye],
                                temp_depth_mask_path,
                                depth_foreground_threshold,
                                fps,
                            )
                            os.replace(temp_depth_mask_path, eye_depth_mask_paths[eye])
                            mark_completed(state, depth_step)
                            save_resume_state(
                                state_path, state, input_video_path, output_stem, config
                            )
                    finally:
                        del depth_model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            pending_sam_eyes = []
            for eye in ("L", "R"):
                sam_step = f"sam_{eye}"
                if is_completed_sam_eye(
                    state,
                    sam_step,
                    eye,
                    eye_green_paths[eye],
                    eye_alpha_paths[eye],
                    (
                        eye_depth_green_paths[eye],
                        eye_depth_raw_green_paths[eye],
                        eye_sam_green_paths[eye],
                        eye_sam_raw_green_paths[eye],
                    ),
                    output_alpha_video,
                    output_instance_videos,
                ):
                    sidecar["eyes"][eye] = state["eyes"][eye]
                else:
                    pending_sam_eyes.append(eye)

            if len(pending_sam_eyes) > 0:
                predictor = create_sam_predictor(
                    sam_model_id,
                    sam_compile,
                )
                try:
                    with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=torch.bfloat16
                    ):
                        for eye in pending_sam_eyes:
                            sam_step = f"sam_{eye}"
                            remove_if_exists(get_temp_path(eye_green_paths[eye]))
                            remove_if_exists(get_temp_path(eye_alpha_paths[eye]))
                            for debug_video_path in (
                                eye_depth_green_paths[eye],
                                eye_depth_raw_green_paths[eye],
                                eye_sam_green_paths[eye],
                                eye_sam_raw_green_paths[eye],
                            ):
                                if debug_video_path is not None:
                                    remove_if_exists(get_temp_path(debug_video_path))
                            eye_data = process_eye_with_sam(
                                predictor,
                                eye_video_paths[eye],
                                input_video_path,
                                get_temp_path(eye_green_paths[eye]),
                                get_temp_path(eye_alpha_paths[eye]),
                                get_optional_temp_path(eye_depth_green_paths[eye]),
                                get_optional_temp_path(eye_depth_raw_green_paths[eye]),
                                get_optional_temp_path(eye_sam_green_paths[eye]),
                                get_optional_temp_path(eye_sam_raw_green_paths[eye]),
                                state,
                                state_path,
                                input_video_path,
                                config,
                                output_dir,
                                output_stem,
                                eye,
                                prompts,
                                fps,
                                sam_chunk_seconds,
                                sam_chunk_overlap_seconds,
                                output_alpha_video,
                                output_instance_videos,
                                sam_mask_close_kernel,
                                sam_mask_border_shift,
                                sam_mask_overlap_gap_fill,
                                qc_frame_interval,
                                qc_area_jump_threshold,
                                greenscreen_crf,
                                greenscreen_preset,
                                depth_mask_paths.get(eye),
                                depth_foreground_threshold,
                                depth_mask_border_shift,
                            )
                            os.replace(
                                get_temp_path(eye_green_paths[eye]),
                                eye_green_paths[eye],
                            )
                            if output_alpha_video:
                                os.replace(
                                    get_temp_path(eye_alpha_paths[eye]),
                                    eye_alpha_paths[eye],
                                )
                            for debug_video_path in (
                                eye_depth_green_paths[eye],
                                eye_depth_raw_green_paths[eye],
                                eye_sam_green_paths[eye],
                                eye_sam_raw_green_paths[eye],
                            ):
                                if debug_video_path is not None:
                                    os.replace(
                                        get_temp_path(debug_video_path),
                                        debug_video_path,
                                    )
                            sidecar["eyes"][eye] = eye_data
                            if eye == "L":
                                sidecar["eyes"][eye]["depth"] = os.path.basename(
                                    retained_depth_path
                                )
                                sidecar["eyes"][eye]["depth_raw"] = os.path.basename(
                                    retained_depth_raw_path
                                )
                                sidecar["eyes"][eye]["sam"] = os.path.basename(
                                    retained_sam_path
                                )
                                sidecar["eyes"][eye]["sam_raw"] = os.path.basename(
                                    retained_sam_raw_path
                                )
                            state.setdefault("eyes", {})[eye] = eye_data
                            mark_completed(state, sam_step)
                            save_resume_state(
                                state_path, state, input_video_path, output_stem, config
                            )
                            print(
                                f"==> eye {eye}: {eye_data['max_instances']} max instances, "
                                f"{len(eye_data['qc_flags'])} QC flags",
                                flush=True,
                            )
                finally:
                    del predictor
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        except Exception as error:
            sidecar["failures"].append(f"SAM 3 processing crashed: {error}")
            print(f"==> SAM 3 processing crashed: {error}", flush=True)
            save_sidecar(sidecar_path, sidecar)
            write_qc_log(qc_log_path, sidecar)
            return

        hstack_videos(
            eye_green_paths["L"], eye_green_paths["R"], greenscreen_path, overwrite=True
        )
        print("==> writing chroma-key metadata", flush=True)
        write_mp4_metadata(greenscreen_path, np.array([0, 255, 0], dtype=np.uint8))
        sidecar["outputs"]["greenscreen"] = os.path.basename(greenscreen_path)
        print(f"==> saved greenscreen composite: {greenscreen_path}", flush=True)

        if output_alpha_video:
            alpha_path = os.path.join(output_dir, f"{output_stem}_alpha.mp4")
            hstack_videos(
                eye_alpha_paths["L"], eye_alpha_paths["R"], alpha_path, overwrite=True
            )
            sidecar["outputs"]["alpha"] = os.path.basename(alpha_path)
            print(f"==> saved alpha video: {alpha_path}", flush=True)

        sidecar["outputs"]["left_eye"] = os.path.basename(retained_eye_path)
        sidecar["outputs"]["left_greenscreen"] = os.path.basename(retained_green_path)
        sidecar["outputs"]["left_depth"] = os.path.basename(retained_depth_path)
        sidecar["outputs"]["left_depth_raw"] = os.path.basename(retained_depth_raw_path)
        sidecar["outputs"]["left_sam"] = os.path.basename(retained_sam_path)
        sidecar["outputs"]["left_sam_raw"] = os.path.basename(retained_sam_raw_path)
        sidecar["outputs"]["state"] = os.path.basename(retained_state_path)
        save_sidecar(sidecar_path, sidecar)
        write_qc_log(qc_log_path, sidecar)
        print(f"==> saved matte sidecar: {sidecar_path}", flush=True)
        if len(sidecar["failures"]) == 0:
            mark_completed(state, "pipeline")
            save_resume_state(state_path, state, input_video_path, output_stem, config)
            pipeline_complete = True
    finally:
        if pipeline_complete and os.path.isdir(work_dir):
            retained_artifacts = (
                (eye_video_paths["L"], retained_eye_path),
                (eye_green_paths["L"], retained_green_path),
                (eye_depth_green_paths["L"], retained_depth_path),
                (eye_depth_raw_green_paths["L"], retained_depth_raw_path),
                (eye_sam_green_paths["L"], retained_sam_path),
                (eye_sam_raw_green_paths["L"], retained_sam_raw_path),
                (state_path, retained_state_path),
            )
            for source_path, destination_path in retained_artifacts:
                temp_destination_path = get_temp_path(destination_path)
                remove_if_exists(temp_destination_path)
                shutil.copy2(source_path, temp_destination_path)
                os.replace(temp_destination_path, destination_path)
            shutil.rmtree(work_dir, ignore_errors=True)


def get_video_properties(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise ValueError("Could not read video FPS")
    if width <= 0 or height <= 0:
        raise ValueError("Could not read video size")
    if frame_count <= 0:
        raise ValueError("Could not read video frame count")
    return fps, width, height, frame_count


def get_sampled_frame_indices(frame_count, num_sampled_frames):
    if num_sampled_frames < 1:
        raise ValueError("num_sampled_frames must be at least 1")
    positions = np.linspace(0, frame_count - 1, num_sampled_frames)
    return np.unique(np.rint(positions).astype(np.int64))


def load_qwen_prompts():
    qwen_prompts_path = g.PLUS_DIR / "greenscreen_prompts.json"
    if not qwen_prompts_path.exists():
        print("greenscreen_prompts.json couldn't be found!.")
        print(
            "Make a copy of plus/greenscreen_prompts_example.json and name it greenscreen_prompts.json."
        )
        print("Exiting...")
        raise FileNotFoundError("greenscreen_prompts.json couldn't be found")

    with open(qwen_prompts_path, "r", encoding="utf-8") as qwen_prompts_file:
        return json.load(qwen_prompts_file)


def parse_prompt_override(prompt_override):
    try:
        prompts = ast.literal_eval(prompt_override)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Could not parse prompt_override: {error}") from error
    if not isinstance(prompts, (list, tuple)):
        raise ValueError("prompt_override must be a Python list of strings")
    prompts = [prompt.strip() for prompt in prompts if isinstance(prompt, str)]
    prompts = [prompt for prompt in prompts if prompt != ""]
    if len(prompts) == 0:
        raise ValueError("prompt_override must contain at least one non-empty string")
    return prompts


def load_resume_state(state_path, input_video_path, output_stem, config):
    if os.path.isfile(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as file:
                state = json.load(file)
        except (json.JSONDecodeError, OSError):
            state = {}
        if resume_state_matches(state, input_video_path, output_stem, config):
            return state
    return {"completed": {}, "eyes": {}}


def resume_state_matches(state, input_video_path, output_stem, config):
    if state.get("input_video_path") != input_video_path:
        return False
    if state.get("output_stem") != output_stem:
        return False
    return state.get("config") == config


def save_resume_state(state_path, state, input_video_path, output_stem, config):
    state["input_video_path"] = input_video_path
    state["output_stem"] = output_stem
    state["config"] = config
    state.setdefault("completed", {})
    state.setdefault("eyes", {})
    temp_state_path = get_temp_path(state_path)
    with open(temp_state_path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, default=str)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_state_path, state_path)


def restore_sidecar_from_state(sidecar, state):
    for key in (
        "include",
        "specific_include",
        "exclude",
        "sam_prompts",
        "prompt_source",
    ):
        if key in state:
            sidecar[key] = state[key]
    sidecar["eyes"].update(state.get("eyes", {}))


def mark_completed(state, step):
    state.setdefault("completed", {})[step] = True


def is_completed(state, step):
    return state.get("completed", {}).get(step) is True


def is_completed_file(state, step, path):
    return is_completed(state, step) and os.path.isfile(path)


def is_completed_sam_eye(
    state,
    step,
    eye,
    green_video_path,
    alpha_video_path,
    debug_video_paths,
    output_alpha_video,
    output_instance_videos,
):
    if output_instance_videos:
        return False
    if not is_completed_file(state, step, green_video_path):
        return False
    if output_alpha_video and not os.path.isfile(alpha_video_path):
        return False
    for debug_video_path in debug_video_paths:
        if debug_video_path is not None and not os.path.isfile(debug_video_path):
            return False
    return eye in state.get("eyes", {})


def get_sam_chunk_record(state, eye, chunk_key):
    return state.get("sam_chunks", {}).get(eye, {}).get(chunk_key)


def set_sam_chunk_record(state, eye, chunk_key, chunk_record):
    state.setdefault("sam_chunks", {}).setdefault(eye, {})[chunk_key] = chunk_record


def get_sam_chunk_paths(
    chunk_dir, range_index, output_alpha_video, output_debug_videos
):
    chunk_base = f"chunk_{range_index:06d}"
    paths = {
        "green": os.path.join(chunk_dir, f"{chunk_base}_green.mp4"),
    }
    if output_alpha_video:
        paths["alpha"] = os.path.join(chunk_dir, f"{chunk_base}_alpha.mp4")
    if output_debug_videos:
        paths["depth"] = os.path.join(chunk_dir, f"{chunk_base}_depth.mp4")
        paths["depth_raw"] = os.path.join(chunk_dir, f"{chunk_base}_depth_raw.mp4")
        paths["sam"] = os.path.join(chunk_dir, f"{chunk_base}_sam.mp4")
        paths["sam_raw"] = os.path.join(chunk_dir, f"{chunk_base}_sam_raw.mp4")
    return paths


def sam_chunk_is_complete(chunk_record, chunk_paths, output_alpha_video):
    if not isinstance(chunk_record, dict):
        return False
    for key, chunk_path in chunk_paths.items():
        if chunk_record.get(key) != chunk_path:
            return False
        if not os.path.isfile(chunk_path):
            return False
    if output_alpha_video and "alpha" not in chunk_record:
        return False
    return True


def get_completed_sample_frames(state, frames_dir, expected_count):
    if not is_completed(state, "sample_frames"):
        return None
    image_paths = state.get("image_paths", [])
    if len(image_paths) != expected_count:
        return None
    for image_path in image_paths:
        if not os.path.isfile(image_path):
            return None
    return image_paths


def get_temp_path(path):
    directory = os.path.dirname(path)
    file_name = os.path.basename(path)
    stem, extension = os.path.splitext(file_name)
    return os.path.join(directory, f".{stem}.tmp{extension}")


def get_optional_temp_path(path):
    if path is None:
        return None
    return get_temp_path(path)


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def extract_sample_frames(
    input_video_path, sampled_frame_indices, eye_width, frames_dir, max_long_side
):
    os.makedirs(frames_dir, exist_ok=True)
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")
    image_paths = []
    try:
        for order, frame_index in enumerate(sampled_frame_indices):
            video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame_bgr = video.read()
            if not success:
                continue
            left_frame = downscale_long_side(frame_bgr[:, :eye_width], max_long_side)
            image_path = os.path.join(frames_dir, f"sample_{order:02d}.jpg")
            cv2.imwrite(image_path, left_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            image_paths.append(image_path)
    finally:
        video.release()
    return image_paths


def downscale_long_side(image, max_long_side):
    height, width = image.shape[:2]
    long_side = max(height, width)
    if max_long_side <= 0 or long_side <= max_long_side:
        return image
    scale = max_long_side / long_side
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def identify_foreground(
    vlm_model_id,
    device,
    image_paths,
    num_vote_runs,
    vlm_temperature,
    vlm_max_new_tokens,
    sidecar,
    qwen_prompts,
):
    model, processor = create_vlm(vlm_model_id, device)
    messages = build_vlm_messages(image_paths, qwen_prompts)
    run_results = []
    try:
        for run_index in range(num_vote_runs):
            try:
                text = run_vlm_once(
                    model, processor, messages, vlm_temperature, vlm_max_new_tokens
                )
                parsed = parse_vlm_json(text)
                if parsed is None:
                    sidecar["failures"].append(
                        f"VLM run {run_index} returned unparseable JSON"
                    )
                run_results.append(parsed)
            except Exception as error:
                sidecar["failures"].append(f"VLM run {run_index} failed: {error}")
                print(f"==> VLM run {run_index} failed: {error}", flush=True)
                run_results.append(None)
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return vote_concepts(run_results)


def create_vlm(vlm_model_id, device):
    print(f"Loading Qwen3-VL model: {vlm_model_id}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        vlm_model_id,
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(vlm_model_id, trust_remote_code=True)
    return model, processor


def build_vlm_messages(image_paths, qwen_prompts):
    content = [{"type": "image", "image": image_path} for image_path in image_paths]
    content.append({"type": "text", "text": qwen_prompts["user_instruction"]})
    return [
        {"role": "system", "content": qwen_prompts["system_prompt"]},
        {"role": "user", "content": content},
    ]


def run_vlm_once(model, processor, messages, vlm_temperature, vlm_max_new_tokens):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=vlm_max_new_tokens,
            do_sample=True,
            temperature=vlm_temperature,
        )
    prompt_length = inputs["input_ids"].shape[1]
    trimmed = generated[:, prompt_length:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def parse_vlm_json(text):
    cleaned = re.sub(r"^```[a-zA-Z]*", "", text.strip()).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for position in range(start, len(cleaned)):
        character = cleaned[position]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                end = position
                break
    if end < 0:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def normalize_concept(phrase):
    text = phrase.strip().lower().replace("'s", "")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = []
    for word in text.split(" "):
        if word in ("the", "a", "an"):
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def vote_concepts(run_results):
    include_representative = {}
    specific_representative = {}
    exclude_representative = {}
    for parsed in run_results:
        if parsed is None:
            continue
        add_concept_list(parsed.get("include", []), include_representative)
        add_concept_list(parsed.get("specific_include", []), specific_representative)
        scene_foreground = parsed.get("scene_foreground", None)
        if isinstance(scene_foreground, str):
            add_concept_list([scene_foreground], specific_representative)
        add_concept_list(parsed.get("exclude", []), exclude_representative)

    include_concepts = list(include_representative.values())
    specific_concepts = list(specific_representative.values())
    exclude_concepts = list(exclude_representative.values())
    return include_concepts, specific_concepts, exclude_concepts


def add_concept_list(phrases, representative):
    if not isinstance(phrases, list):
        return
    for phrase in phrases:
        if not isinstance(phrase, str):
            continue
        key = normalize_concept(phrase)
        if key == "":
            continue
        representative.setdefault(key, phrase.strip())


def build_sam_prompts(
    include_concepts, specific_include_concepts, sam_prompt_groups, qwen_prompts
):
    candidates = []
    for concept in include_concepts:
        candidates.append(concept)
    for concept in specific_include_concepts:
        candidates.append(concept)
    for concept in qwen_prompts["common_concepts"]:
        if prompt_matches_scene(concept, include_concepts, specific_include_concepts):
            candidates.append(concept)
    if len(candidates) == 0:
        candidates.extend(qwen_prompts["common_concepts"][:2])

    prompts = []
    seen = set()
    for concept in candidates:
        key = normalize_concept(concept)
        if key == "" or key in seen:
            continue
        seen.add(key)
        prompts.append(concept.strip())

    if sam_prompt_groups > 0:
        prompts = prompts[:sam_prompt_groups]
    return prompts


def prompt_matches_scene(concept, include_concepts, specific_include_concepts):
    text = normalize_concept(" ".join(include_concepts + specific_include_concepts))
    concept_key = normalize_concept(concept)
    for word in concept_key.split(" "):
        if len(word) >= 4 and word in text:
            return True
    return False


def create_depth_model():
    print("Loading ZoeD_Any_N depth model", flush=True)
    depth_model = ZoeDepthModel("ZoeD_Any_N")
    depth_model.load(gpu=0)
    return depth_model


def compute_eye_depth_mask_video(
    depth_model,
    eye_video_path,
    depth_mask_path,
    depth_foreground_threshold,
    fps,
):
    video = cv2.VideoCapture(eye_video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open cropped eye video: {eye_video_path}")
    try:
        _, width, height, _ = get_video_properties(video)
        writer = g.RawVideoWriter(depth_mask_path, width, height, fps, codec="ffv1")
        try:
            while True:
                success, frame_bgr = video.read()
                if not success:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(
                    frame_rgb.astype(np.float32) / 255.0
                ).permute(2, 0, 1)
                depth = depth_model.infer(frame_tensor.to(depth_model.device))
                depth = depth_model.minmax_normalize_chw(depth)
                depth_np = depth.squeeze(0).cpu().numpy()
                mask = (depth_np >= 1.0 - depth_foreground_threshold).astype(np.uint8)
                if mask.shape != (height, width):
                    mask = cv2.resize(
                        mask, (width, height), interpolation=cv2.INTER_NEAREST
                    )
                writer.write(np.repeat(mask[:, :, None] * 255, 3, axis=2))
        finally:
            writer.close()
    finally:
        video.release()


def apply_depth_mask(
    alpha,
    depth_mask,
    depth_foreground_threshold,
    depth_mask_border_shift=0,
):
    if depth_foreground_threshold <= 0 or depth_mask is None:
        return alpha
    mask = depth_mask
    if depth_mask_border_shift != 0:
        shift = abs(int(depth_mask_border_shift))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (shift * 2 + 1, shift * 2 + 1)
        )
        if depth_mask_border_shift > 0:
            mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            mask = cv2.erode(mask, kernel, iterations=1)
    return np.maximum(alpha, mask.astype(np.float32) / 255.0)


def create_sam_predictor(
    sam_model_id,
    compiled,
):
    print(f"Loading SAM 3 model: {sam_model_id}", flush=True)

    dtype = torch.bfloat16
    model = Sam3VideoModel.from_pretrained(sam_model_id).to("cuda", dtype=dtype)
    if compiled:
        model = torch.compile(model)
    model.eval()
    processor = Sam3VideoProcessor.from_pretrained(sam_model_id)
    return {
        "model": model,
        "processor": processor,
        "device": "cuda",
        "dtype": dtype,
    }


def crop_eye_video(
    input_video_path,
    eye,
    output_video_path,
    max_long_side,
    crf,
    preset,
):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    if eye == "L":
        crop = "crop=iw/2:ih:0:0"
    else:
        crop = "crop=iw/2:ih:iw/2:0"
    filters = [crop]
    if max_long_side > 0:
        filters.append(
            f"scale='if(gt(max(iw,ih),{max_long_side}),if(gte(iw,ih),{max_long_side},-2),iw)':"
            f"'if(gt(max(iw,ih),{max_long_side}),if(gte(iw,ih),-2,{max_long_side}),ih)'"
        )
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_video_path,
        "-vf",
        ",".join(filters),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        output_video_path,
    ]
    g.run_command(command)


def process_eye_with_sam(
    predictor,
    sam_eye_video_path,
    input_video_path,
    green_video_path,
    alpha_video_path,
    depth_green_video_path,
    depth_raw_green_video_path,
    sam_green_video_path,
    sam_raw_green_video_path,
    state,
    state_path,
    resume_input_video_path,
    config,
    output_dir,
    output_stem,
    eye,
    prompts,
    fps,
    sam_chunk_seconds,
    sam_chunk_overlap_seconds,
    output_alpha_video,
    output_instance_videos,
    sam_mask_close_kernel,
    sam_mask_border_shift,
    sam_mask_overlap_gap_fill,
    qc_frame_interval,
    qc_area_jump_threshold,
    greenscreen_crf=18,
    greenscreen_preset="veryfast",
    depth_mask_path=None,
    depth_foreground_threshold=0.0,
    depth_mask_border_shift=0,
):
    video = cv2.VideoCapture(sam_eye_video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open cropped eye video: {sam_eye_video_path}")
    try:
        _, _, _, frame_count = get_video_properties(video)
    finally:
        video.release()

    input_video = cv2.VideoCapture(input_video_path)
    if not input_video.isOpened():
        raise ValueError(f"Could not open input video: {input_video_path}")
    try:
        _, input_width, height, input_frame_count = get_video_properties(input_video)
    finally:
        input_video.release()
    if input_width % 2 != 0:
        raise ValueError(f"SBS video width must be even, got: {input_width}")
    if input_frame_count != frame_count:
        raise ValueError("SAM eye video frame count does not match input video")
    width = input_width // 2
    model = predictor["model"]
    processor = predictor["processor"]
    device = predictor["device"]
    dtype = predictor["dtype"]

    instance_writers = {}
    instance_records = {}
    previous_area = None
    previous_present_count = None
    max_instances = 0
    qc_flags = []

    ranges = get_sam_chunk_ranges(
        frame_count, fps, sam_chunk_seconds, sam_chunk_overlap_seconds
    )
    chunk_dir = os.path.join(os.path.dirname(green_video_path), f"sam_{eye}_chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    output_debug_videos = all(
        video_path is not None
        for video_path in (
            depth_green_video_path,
            depth_raw_green_video_path,
            sam_green_video_path,
            sam_raw_green_video_path,
        )
    )
    chunk_green_paths = []
    chunk_alpha_paths = []
    chunk_debug_paths = {
        "depth": [],
        "depth_raw": [],
        "sam": [],
        "sam_raw": [],
    }

    for range_index, (chunk_start, chunk_end, output_start) in enumerate(ranges):
        chunk_key = str(range_index)
        chunk_record = get_sam_chunk_record(state, eye, chunk_key)
        chunk_paths = get_sam_chunk_paths(
            chunk_dir, range_index, output_alpha_video, output_debug_videos
        )
        chunk_green_paths.append(chunk_paths["green"])
        if output_alpha_video:
            chunk_alpha_paths.append(chunk_paths["alpha"])
        if output_debug_videos:
            for debug_name in chunk_debug_paths:
                chunk_debug_paths[debug_name].append(chunk_paths[debug_name])
        if not output_instance_videos and sam_chunk_is_complete(
            chunk_record, chunk_paths, output_alpha_video
        ):
            max_instances = max(max_instances, chunk_record.get("max_instances", 0))
            qc_flags.extend(chunk_record.get("qc_flags", []))
            previous_area = chunk_record.get("last_area", previous_area)
            previous_present_count = chunk_record.get(
                "last_present_count", previous_present_count
            )
            continue

        for chunk_path in chunk_paths.values():
            remove_if_exists(get_temp_path(chunk_path))

        green_writer = g.RawVideoWriter(
            get_temp_path(chunk_paths["green"]),
            width,
            height,
            fps,
            codec="libx264",
            crf=greenscreen_crf,
            preset=greenscreen_preset,
            pixel_format="yuv420p",
        )
        alpha_writer = None
        if output_alpha_video:
            alpha_writer = g.RawVideoWriter(
                get_temp_path(chunk_paths["alpha"]),
                width,
                height,
                fps,
                codec="libx264",
                crf=12,
                preset="veryfast",
                pixel_format="yuv420p",
            )
        debug_writers = {}
        if output_debug_videos:
            for debug_name in chunk_debug_paths:
                debug_writers[debug_name] = g.RawVideoWriter(
                    get_temp_path(chunk_paths[debug_name]),
                    width,
                    height,
                    fps,
                    codec="libx264",
                    crf=greenscreen_crf,
                    preset=greenscreen_preset,
                    pixel_format="yuv420p",
                )
        chunk_qc_flags = []
        chunk_max_instances = 0
        try:
            print(
                f"==> eye {eye}: SAM frames {chunk_start}-{chunk_end - 1}, "
                f"writing from {output_start}",
                flush=True,
            )
            video_frames = load_video_frame_range(
                sam_eye_video_path, chunk_start, chunk_end
            )
            output_frames = load_eye_frame_range(
                input_video_path, eye, chunk_start, chunk_end, width
            )
            if len(output_frames) != len(video_frames):
                raise ValueError("SAM eye video frames do not match input video frames")
            inference_session = processor.init_video_session(
                video=video_frames,
                inference_device=device,
                processing_device=device,
                video_storage_device="cpu",
                dtype=dtype,
            )
            for prompt in prompts:
                inference_session = processor.add_text_prompt(
                    inference_session=inference_session,
                    text=prompt,
                )
            depth_video = open_depth_mask_video(depth_mask_path, output_start)
            actual_chunk_end = chunk_start + len(video_frames)
            next_output_frame = output_start
            try:
                for model_outputs in model.propagate_in_video_iterator(
                    inference_session=inference_session,
                    max_frame_num_to_track=len(video_frames) - 1,
                ):
                    local_frame_index = int(model_outputs.frame_idx)
                    frame_index = chunk_start + local_frame_index
                    if frame_index < output_start:
                        continue
                    while next_output_frame < frame_index:
                        skipped_local_frame_index = next_output_frame - chunk_start
                        skipped_frame_rgb = output_frames[skipped_local_frame_index]
                        depth_mask = read_depth_mask(depth_video, height, width)
                        empty_sam_alpha = np.zeros((height, width), dtype=np.float32)
                        empty_alpha = apply_depth_mask(
                            empty_sam_alpha,
                            depth_mask,
                            depth_foreground_threshold,
                            depth_mask_border_shift,
                        )
                        write_debug_green_frames(
                            debug_writers,
                            skipped_frame_rgb,
                            depth_mask,
                            depth_foreground_threshold,
                            depth_mask_border_shift,
                            empty_sam_alpha,
                            empty_sam_alpha,
                        )
                        write_green_and_alpha(
                            green_writer, alpha_writer, skipped_frame_rgb, empty_alpha
                        )
                        next_output_frame += 1
                    frame_rgb = output_frames[local_frame_index]
                    depth_mask = read_depth_mask(depth_video, height, width)
                    processed_outputs = processor.postprocess_outputs(
                        inference_session, model_outputs
                    )
                    masks = tensor_to_numpy(processed_outputs.get("masks", None))
                    object_ids = tensor_to_list(processed_outputs.get("object_ids", []))
                    sam_alpha, sam_raw_alpha, present_count = combine_masks(
                        masks,
                        height,
                        width,
                        sam_mask_close_kernel,
                        sam_mask_border_shift,
                        sam_mask_overlap_gap_fill,
                    )
                    write_debug_green_frames(
                        debug_writers,
                        frame_rgb,
                        depth_mask,
                        depth_foreground_threshold,
                        depth_mask_border_shift,
                        sam_alpha,
                        sam_raw_alpha,
                    )
                    combined_alpha = sam_alpha
                    combined_alpha = apply_depth_mask(
                        combined_alpha,
                        depth_mask,
                        depth_foreground_threshold,
                        depth_mask_border_shift,
                    )
                    max_instances = max(max_instances, present_count)
                    chunk_max_instances = max(chunk_max_instances, present_count)
                    write_green_and_alpha(
                        green_writer, alpha_writer, frame_rgb, combined_alpha
                    )
                    if output_instance_videos and masks is not None:
                        write_instance_frames(
                            instance_writers,
                            instance_records,
                            output_dir,
                            output_stem,
                            eye,
                            frame_rgb,
                            masks,
                            object_ids,
                            width,
                            height,
                            fps,
                        )
                    if frame_index % qc_frame_interval == 0:
                        area = float((combined_alpha > 0.05).mean())
                        present_count = len(object_ids) if object_ids is not None else 0
                        collect_qc_flags(
                            frame_index,
                            area,
                            present_count,
                            previous_area,
                            previous_present_count,
                            qc_area_jump_threshold,
                            chunk_qc_flags,
                        )
                        previous_area = area
                        previous_present_count = present_count
                    next_output_frame = frame_index + 1
                while next_output_frame < actual_chunk_end:
                    skipped_local_frame_index = next_output_frame - chunk_start
                    skipped_frame_rgb = output_frames[skipped_local_frame_index]
                    depth_mask = read_depth_mask(depth_video, height, width)
                    empty_sam_alpha = np.zeros((height, width), dtype=np.float32)
                    empty_alpha = apply_depth_mask(
                        empty_sam_alpha,
                        depth_mask,
                        depth_foreground_threshold,
                        depth_mask_border_shift,
                    )
                    write_debug_green_frames(
                        debug_writers,
                        skipped_frame_rgb,
                        depth_mask,
                        depth_foreground_threshold,
                        depth_mask_border_shift,
                        empty_sam_alpha,
                        empty_sam_alpha,
                    )
                    write_green_and_alpha(
                        green_writer, alpha_writer, skipped_frame_rgb, empty_alpha
                    )
                    next_output_frame += 1
            finally:
                if depth_video is not None:
                    depth_video.release()
                del inference_session
                del video_frames
                del output_frames
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            green_writer.close()
            if alpha_writer is not None:
                alpha_writer.close()
            for debug_writer in debug_writers.values():
                debug_writer.close()

        os.replace(get_temp_path(chunk_paths["green"]), chunk_paths["green"])
        if output_alpha_video:
            os.replace(get_temp_path(chunk_paths["alpha"]), chunk_paths["alpha"])
        if output_debug_videos:
            for debug_name in chunk_debug_paths:
                os.replace(
                    get_temp_path(chunk_paths[debug_name]), chunk_paths[debug_name]
                )
        qc_flags.extend(chunk_qc_flags)
        chunk_record = {
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "output_start": output_start,
            "max_instances": chunk_max_instances,
            "qc_flags": chunk_qc_flags,
            "last_area": previous_area,
            "last_present_count": previous_present_count,
        }
        chunk_record.update(chunk_paths)
        set_sam_chunk_record(
            state,
            eye,
            chunk_key,
            chunk_record,
        )
        save_resume_state(
            state_path, state, resume_input_video_path, output_stem, config
        )

    concat_videos(chunk_green_paths, green_video_path)
    if output_alpha_video:
        concat_videos(chunk_alpha_paths, alpha_video_path)
    if output_debug_videos:
        debug_video_paths = {
            "depth": depth_green_video_path,
            "depth_raw": depth_raw_green_video_path,
            "sam": sam_green_video_path,
            "sam_raw": sam_raw_green_video_path,
        }
        for debug_name, debug_video_path in debug_video_paths.items():
            concat_videos(chunk_debug_paths[debug_name], debug_video_path)

    for writer in instance_writers.values():
        writer.close()
    gc.collect()

    return {
        "max_instances": max_instances,
        "instances": list(instance_records.values()),
        "qc_flags": qc_flags,
    }


def get_sam_chunk_ranges(
    frame_count, fps, sam_chunk_seconds, sam_chunk_overlap_seconds
):
    if sam_chunk_seconds <= 0:
        return [(0, frame_count, 0)]
    chunk_frames = max(1, int(round(sam_chunk_seconds * fps)))
    overlap_frames = max(0, int(round(sam_chunk_overlap_seconds * fps)))
    overlap_frames = max(overlap_frames, SAM_HOTSTART_FRAMES + 5)
    if overlap_frames >= chunk_frames:
        raise ValueError(
            "sam_chunk_overlap_seconds must be less than sam_chunk_seconds"
        )

    ranges = []
    output_start = 0
    while output_start < frame_count:
        chunk_start = max(0, output_start - overlap_frames)
        chunk_end = min(frame_count, output_start + chunk_frames)
        ranges.append((chunk_start, chunk_end, output_start))
        output_start = chunk_end
    return ranges


def load_video_frame_range(video_path, start_frame_index, end_frame_index):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open cropped eye video: {video_path}")
    try:
        frames = []
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
        for _ in range(start_frame_index, end_frame_index):
            success, frame_bgr = video.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        video.release()
    if len(frames) == 0:
        raise ValueError(f"Could not read cropped eye video frames: {video_path}")
    return frames


def load_eye_frame_range(
    input_video_path, eye, start_frame_index, end_frame_index, eye_width
):
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open input video: {input_video_path}")
    try:
        frames = []
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
        for _ in range(start_frame_index, end_frame_index):
            success, frame_bgr = video.read()
            if not success:
                break
            if eye == "L":
                eye_frame_bgr = frame_bgr[:, :eye_width]
            else:
                eye_frame_bgr = frame_bgr[:, eye_width:]
            frames.append(cv2.cvtColor(eye_frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        video.release()
    if len(frames) == 0:
        raise ValueError(f"Could not read input video frames: {input_video_path}")
    return frames


def open_depth_mask_video(depth_mask_path, start_frame_index):
    if depth_mask_path is None:
        return None
    video = cv2.VideoCapture(depth_mask_path)
    if not video.isOpened():
        raise ValueError(f"Could not open depth mask video: {depth_mask_path}")
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    return video


def read_depth_mask(video, height, width):
    if video is None:
        return None
    success, frame_bgr = video.read()
    if not success:
        raise ValueError("Could not read depth mask frame")
    mask = frame_bgr[:, :, 0]
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def write_green_and_alpha(green_writer, alpha_writer, frame_rgb, alpha):
    green_writer.write(composite_green(frame_rgb, alpha))
    if alpha_writer is None:
        return
    alpha_u8 = np.round(alpha * 255.0).clip(0, 255).astype(np.uint8)
    alpha_writer.write(cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2RGB))


def write_debug_green_frames(
    debug_writers,
    frame_rgb,
    depth_mask,
    depth_foreground_threshold,
    depth_mask_border_shift,
    sam_alpha,
    sam_raw_alpha,
):
    if len(debug_writers) == 0:
        return
    empty_alpha = np.zeros(sam_alpha.shape, dtype=np.float32)
    depth_raw_alpha = apply_depth_mask(
        empty_alpha, depth_mask, depth_foreground_threshold
    )
    depth_alpha = apply_depth_mask(
        empty_alpha,
        depth_mask,
        depth_foreground_threshold,
        depth_mask_border_shift,
    )
    debug_writers["depth"].write(composite_green(frame_rgb, depth_alpha))
    debug_writers["depth_raw"].write(composite_green(frame_rgb, depth_raw_alpha))
    debug_writers["sam"].write(composite_green(frame_rgb, sam_alpha))
    debug_writers["sam_raw"].write(composite_green(frame_rgb, sam_raw_alpha))


def tensor_to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_to_list(value):
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def combine_masks(
    masks,
    height,
    width,
    sam_mask_close_kernel,
    sam_mask_border_shift,
    sam_mask_overlap_gap_fill,
):
    if masks is None:
        empty_alpha = np.zeros((height, width), dtype=np.float32)
        return empty_alpha, empty_alpha, 0
    masks = np.asarray(masks)
    if masks.size == 0:
        empty_alpha = np.zeros((height, width), dtype=np.float32)
        return empty_alpha, empty_alpha, 0
    if masks.ndim == 2:
        masks = masks[None]
    combined = np.zeros((height, width), dtype=np.uint8)
    instance_masks = []
    present_count = 0
    for mask in masks:
        mask_2d = np.squeeze(mask)
        if mask_2d.ndim != 2:
            continue
        if mask_2d.shape[:2] != (height, width):
            mask_2d = cv2.resize(
                mask_2d.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_u8 = (mask_2d > 0).astype(np.uint8) * 255
        if mask_u8.max() == 0:
            continue
        present_count += 1
        instance_masks.append(mask_u8)
        combined = np.maximum(combined, mask_u8)

    raw_combined = combined.copy()
    combined = fill_overlap_gaps(combined, instance_masks, sam_mask_overlap_gap_fill)
    combined = postprocess_mask(combined, sam_mask_close_kernel, sam_mask_border_shift)
    return (
        combined.astype(np.float32) / 255.0,
        raw_combined.astype(np.float32) / 255.0,
        present_count,
    )


def fill_overlap_gaps(combined, instance_masks, sam_mask_overlap_gap_fill):
    if sam_mask_overlap_gap_fill <= 0 or len(instance_masks) < 2:
        return combined

    kernel_size = int(sam_mask_overlap_gap_fill) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    overlap_count = np.zeros(combined.shape, dtype=np.uint16)
    for instance_mask in instance_masks:
        dilated = cv2.dilate(instance_mask, kernel, iterations=1)
        overlap_count += (dilated > 0).astype(np.uint16)

    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    gap_mask = (combined == 0) & (overlap_count >= 2) & (closed > 0)
    if np.any(gap_mask):
        combined = combined.copy()
        combined[gap_mask] = 255
    return combined


def postprocess_mask(mask_u8, sam_mask_close_kernel, sam_mask_border_shift):
    if sam_mask_close_kernel > 1:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (sam_mask_close_kernel, sam_mask_close_kernel)
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)
    if sam_mask_border_shift != 0:
        border_shift = abs(int(sam_mask_border_shift))
        border_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (border_shift * 2 + 1, border_shift * 2 + 1)
        )
        if sam_mask_border_shift > 0:
            mask_u8 = cv2.dilate(mask_u8, border_kernel, iterations=1)
        else:
            mask_u8 = cv2.erode(mask_u8, border_kernel, iterations=1)
    return mask_u8


def composite_green(frame_rgb, alpha):
    alpha = alpha[:, :, None].astype(np.float32)
    frame = frame_rgb.astype(np.float32)
    green = np.array([0.0, 255.0, 0.0], dtype=np.float32).reshape(1, 1, 3)
    composite = frame * alpha + green * (1.0 - alpha)
    return composite.clip(0, 255).astype(np.uint8)


def write_instance_frames(
    instance_writers,
    instance_records,
    output_dir,
    video_stem,
    eye,
    frame_rgb,
    masks,
    object_ids,
    width,
    height,
    fps,
):
    masks = np.asarray(masks)
    for position, object_id in enumerate(object_ids):
        object_id = int(object_id)
        if object_id not in instance_writers:
            output_path = os.path.join(
                output_dir, f"{video_stem}_{eye}_obj{object_id:03d}.mov"
            )
            instance_writers[object_id] = g.RawAlphaVideoWriter(
                output_path, width, height, fps
            )
            instance_records[object_id] = {
                "id": object_id,
                "concept": "SAM 3 object",
                "output": os.path.basename(output_path),
            }
        mask = np.squeeze(masks[position])
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        alpha16 = (mask > 0).astype(np.uint16) * 65535
        rgba = np.dstack([frame_rgb.astype(np.uint16) * 257, alpha16])
        instance_writers[object_id].write(rgba)


def collect_qc_flags(
    frame_index,
    area,
    present_count,
    previous_area,
    previous_present_count,
    qc_area_jump_threshold,
    qc_flags,
):
    if previous_present_count is not None and present_count != previous_present_count:
        qc_flags.append(
            {
                "frame_index": frame_index,
                "type": "instance_count_change",
                "from": previous_present_count,
                "to": present_count,
            }
        )
    if previous_area is None or previous_area <= 0.0:
        return
    if abs(area - previous_area) / previous_area > qc_area_jump_threshold:
        qc_flags.append(
            {
                "frame_index": frame_index,
                "type": "combined_area_jump",
                "from_area": previous_area,
                "to_area": area,
            }
        )


def hstack_videos(left_path, right_path, output_path, overwrite=False):
    if os.path.isfile(output_path) and not overwrite:
        return
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        left_path,
        "-i",
        right_path,
        "-filter_complex",
        "[0:v][1:v]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "12",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    g.run_command(command)


def concat_videos(input_paths, output_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    directory = os.path.dirname(output_path) or "."
    file_name = os.path.basename(output_path)
    list_path = os.path.join(directory, f".{os.path.splitext(file_name)[0]}_concat.txt")
    with open(list_path, "w", encoding="utf-8") as file:
        for input_path in input_paths:
            file.write(f"file '{os.path.abspath(input_path)}'\n")
    command = [
        ffmpeg_path,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        output_path,
    ]
    try:
        g.run_command(command)
    finally:
        remove_if_exists(list_path)


def get_hex_color(green_color):
    return "#{:02X}{:02X}{:02X}".format(
        int(green_color[0]), int(green_color[1]), int(green_color[2])
    )


def get_chroma_key_metadata(green_color):
    hex_color = get_hex_color(green_color)
    metadata = [
        ("stereo_mode", "left_right"),
        ("chroma_key", "true"),
        ("chroma_key_color", hex_color),
        ("greenscreen", "true"),
        ("greenscreen_color", hex_color),
        ("passthrough_chroma_key", hex_color),
        ("com.oculus.vr.chroma_key", hex_color),
        ("com.meta.vr.chroma_key", hex_color),
        ("com.deovr.chroma_key", hex_color),
    ]
    return metadata


def write_mp4_metadata(video_path, green_color):
    metadata = get_chroma_key_metadata(green_color)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    directory = os.path.dirname(video_path) or "."
    file_name = os.path.basename(video_path)

    with tempfile.NamedTemporaryFile(
        prefix=f".{os.path.splitext(file_name)[0]}_metadata_",
        suffix=".mp4",
        dir=directory,
        delete=False,
    ) as temp_file:
        temp_output_path = temp_file.name

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        video_path,
        "-map",
        "0",
        "-c",
        "copy",
        "-movflags",
        "use_metadata_tags",
    ]

    for key, value in metadata:
        command.extend(["-metadata", f"{key}={value}"])

    command.append(temp_output_path)

    try:
        g.run_command(command)
        os.replace(temp_output_path, video_path)
    finally:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


def save_sidecar(sidecar_path, sidecar):
    with open(sidecar_path, "w", encoding="utf-8") as file:
        json.dump(sidecar, file, indent=2, default=str)


def write_qc_log(qc_log_path, sidecar):
    lines = [f"QC report for {sidecar['input_video_path']}"]
    for failure in sidecar["failures"]:
        lines.append(f"FAILURE: {failure}")
    for eye, eye_data in sidecar["eyes"].items():
        lines.append(f"eye {eye}: {eye_data['max_instances']} max instances")
        for flag in eye_data["qc_flags"]:
            lines.append(f"  {eye} {flag}")
    with open(qc_log_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    Fire(main)
