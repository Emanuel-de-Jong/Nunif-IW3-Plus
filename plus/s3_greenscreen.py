import os
import gc
import re
import json
import math
import glob
import shlex
import shutil
import subprocess
import cv2
import torch
import numpy as np
import imageio_ffmpeg
import plus.global_params as g
from fire import Fire
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

SYSTEM_PROMPT = (
    "You are a VFX matting assistant. Respond with strict, minified JSON only. "
    "No prose, no markdown, no code fences."
)

USER_INSTRUCTION = (
    "The attached images are frames sampled evenly from a single continuous video "
    "shot with a fixed cast of foreground subjects. These objects will be cut out and "
    "placed on a different background, so decide what should and should not be "
    "included as if you were compositing this shot for VFX. "
    "Return JSON with exactly two keys. "
    '"include" is a list of short foreground concept phrases that must survive onto '
    'the new background, for example "the girl in the red coat" or "the brown dog". '
    '"exclude" is a list of things that must NOT be included, and it must always be '
    "populated: actively reason about ambiguous cases such as cast or contact shadows, "
    "reflections, translucent or particle effects, sky, and background scenery. "
    "Use one concept phrase per distinct kind of subject."
)


def main(
    input_video_path: str,
    output_dir: str = None,
    vlm_model_id: str = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
    num_sampled_frames: int = 7,
    num_vote_runs: int = 3,
    vlm_temperature: float = 0.3,
    vlm_max_new_tokens: int = 512,
    sam2matting_repo_dir: str = None,
    qc_frame_interval: int = 15,
    qc_area_jump_threshold: float = 0.40,
    overwrite: bool = False,
):
    device = "cuda"

    if sam2matting_repo_dir is None:
        sam2matting_repo_dir = str(g.PLUS_DIR / "SAM2Matting")
    sam_matting_script = str(g.PLUS_DIR / "sam_matting.py")
    sam2matting_launcher = "conda run --no-capture-output -n sam2matting python"

    video_dir = os.path.dirname(os.path.abspath(input_video_path))
    video_stem = os.path.splitext(os.path.basename(input_video_path))[0]
    if output_dir is None:
        output_dir = os.path.join(video_dir, "plus", f"{video_stem}_matte")

    sidecar_path = os.path.join(output_dir, f"{video_stem}_matte.json")
    qc_log_path = os.path.join(output_dir, f"{video_stem}_qc.log")

    if g.should_skip_output(sidecar_path, overwrite):
        return
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
    work_dir = os.path.join(output_dir, f".{video_stem}_work")
    os.makedirs(work_dir, exist_ok=True)

    config = {
        "vlm_model_id": vlm_model_id,
        "num_sampled_frames": num_sampled_frames,
        "num_vote_runs": num_vote_runs,
        "vlm_temperature": vlm_temperature,
        "vlm_max_new_tokens": vlm_max_new_tokens,
        "sam2matting_repo_dir": sam2matting_repo_dir,
        "qc_frame_interval": qc_frame_interval,
        "qc_area_jump_threshold": qc_area_jump_threshold,
        "device": device,
    }
    sidecar = {
        "input_video_path": input_video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "config": config,
        "sampled_frame_indices": [],
        "include": [],
        "exclude": [],
        "eyes": {},
        "failures": [],
    }

    try:
        sampled_frame_indices = get_sampled_frame_indices(
            frame_count, num_sampled_frames
        )
        sidecar["sampled_frame_indices"] = [
            int(index) for index in sampled_frame_indices
        ]
        frames_dir = os.path.join(work_dir, "frames")
        image_paths = extract_sample_frames(
            input_video_path, sampled_frame_indices, eye_width, frames_dir
        )
        if len(image_paths) == 0:
            sidecar["failures"].append("frame sampling produced no frames")
            print("==> frame sampling produced no frames", flush=True)
            save_sidecar(sidecar_path, sidecar)
            write_qc_log(qc_log_path, sidecar)
            return

        print(f"==> sampled {len(image_paths)} frames for VLM", flush=True)
        include_concepts, exclude_concepts = identify_foreground(
            vlm_model_id,
            device,
            image_paths,
            num_vote_runs,
            vlm_temperature,
            vlm_max_new_tokens,
            sidecar,
        )
        sidecar["include"] = include_concepts
        sidecar["exclude"] = exclude_concepts
        print(f"==> voted include concepts: {include_concepts}", flush=True)
        print(f"==> voted exclude concepts: {exclude_concepts}", flush=True)

        if len(include_concepts) == 0:
            sidecar["failures"].append("no foreground objects detected")
            print("==> no foreground objects detected", flush=True)
            save_sidecar(sidecar_path, sidecar)
            write_qc_log(qc_log_path, sidecar)
            return

        if not os.path.isdir(sam2matting_repo_dir):
            sidecar["failures"].append(
                f"SAM2Matting repo not found: {sam2matting_repo_dir}; skipped matting stage"
            )
            print(
                f"==> SAM2Matting repo not found: {sam2matting_repo_dir}; "
                "skipping matting stage",
                flush=True,
            )
            save_sidecar(sidecar_path, sidecar)
            write_qc_log(qc_log_path, sidecar)
            return

        for eye in ("L", "R"):
            try:
                instance_records, qc_flags = process_eye(
                    eye,
                    input_video_path,
                    work_dir,
                    output_dir,
                    video_stem,
                    fps,
                    include_concepts,
                    sam2matting_launcher,
                    sam2matting_repo_dir,
                    sam_matting_script,
                    qc_frame_interval,
                    qc_area_jump_threshold,
                )
                sidecar["eyes"][eye] = {
                    "instances": instance_records,
                    "qc_flags": qc_flags,
                }
                print(
                    f"==> eye {eye}: {len(instance_records)} instances, "
                    f"{len(qc_flags)} QC flags",
                    flush=True,
                )
            except Exception as error:
                sidecar["failures"].append(
                    f"SAM2Matting crashed for eye {eye}: {error}"
                )
                sidecar["eyes"][eye] = {"instances": [], "qc_flags": []}
                print(f"==> SAM2Matting crashed for eye {eye}: {error}", flush=True)

        save_sidecar(sidecar_path, sidecar)
        write_qc_log(qc_log_path, sidecar)
        print(f"==> saved matte sidecar: {sidecar_path}", flush=True)
    finally:
        if os.path.isdir(work_dir):
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


def extract_sample_frames(
    input_video_path, sampled_frame_indices, eye_width, frames_dir
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
            left_frame = frame_bgr[:, :eye_width]
            image_path = os.path.join(frames_dir, f"sample_{order:02d}.jpg")
            cv2.imwrite(image_path, left_frame)
            image_paths.append(image_path)
    finally:
        video.release()
    return image_paths


def identify_foreground(
    vlm_model_id,
    device,
    image_paths,
    num_vote_runs,
    vlm_temperature,
    vlm_max_new_tokens,
    sidecar,
):
    model, processor = create_vlm(vlm_model_id, device)
    messages = build_vlm_messages(image_paths)
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
    return vote_concepts(run_results, num_vote_runs)


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


def build_vlm_messages(image_paths):
    content = [{"type": "image", "image": image_path} for image_path in image_paths]
    content.append({"type": "text", "text": USER_INSTRUCTION})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
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
    text = re.sub(r"[^a-z0-9 ]", " ", phrase.strip().lower())
    text = re.sub(r"\s+", " ", text).strip()
    for article in ("the ", "a ", "an "):
        if text.startswith(article):
            text = text[len(article) :]
    return text


def vote_concepts(run_results, num_vote_runs):
    include_counts = {}
    include_representative = {}
    exclude_representative = {}
    for parsed in run_results:
        if parsed is None:
            continue
        seen_in_run = set()
        for phrase in parsed.get("include", []):
            if not isinstance(phrase, str):
                continue
            key = normalize_concept(phrase)
            if key == "" or key in seen_in_run:
                continue
            seen_in_run.add(key)
            include_counts[key] = include_counts.get(key, 0) + 1
            include_representative.setdefault(key, phrase.strip())
        for phrase in parsed.get("exclude", []):
            if not isinstance(phrase, str):
                continue
            key = normalize_concept(phrase)
            if key == "":
                continue
            exclude_representative.setdefault(key, phrase.strip())

    threshold = math.ceil(num_vote_runs / 2)
    include_concepts = [
        include_representative[key]
        for key, count in include_counts.items()
        if count >= threshold
    ]
    exclude_concepts = list(exclude_representative.values())
    return include_concepts, exclude_concepts


def process_eye(
    eye,
    input_video_path,
    work_dir,
    output_dir,
    video_stem,
    fps,
    include_concepts,
    sam2matting_launcher,
    sam2matting_repo_dir,
    sam_matting_script,
    qc_frame_interval,
    qc_area_jump_threshold,
):
    eye_frames_dir = os.path.join(work_dir, f"frames_{eye}")
    extract_eye_frames(input_video_path, eye, eye_frames_dir)

    eye_output_dir = os.path.join(work_dir, f"sam_{eye}")
    run_sam_matting(
        sam2matting_launcher,
        sam2matting_repo_dir,
        sam_matting_script,
        eye_frames_dir,
        include_concepts,
        eye_output_dir,
    )

    instances = collect_instance_alpha_sequences(eye_output_dir)
    if len(instances) == 0:
        raise RuntimeError(f"SAM2Matting produced no instances for eye {eye}")

    output_mov_paths = []
    instance_records = []
    for instance in instances:
        concept_label = resolve_concept_label(instance["concept"], include_concepts)
        label = slugify_concept(concept_label)
        filename = f"{video_stem}_{eye}_inst{instance['id']:02d}_{label}.mov"
        output_mov_paths.append(os.path.join(output_dir, filename))
        instance_records.append(
            {"id": instance["id"], "concept": concept_label, "output": filename}
        )

    qc_flags = render_and_qc_eye(
        eye_frames_dir,
        instances,
        output_mov_paths,
        fps,
        qc_frame_interval,
        qc_area_jump_threshold,
    )
    return instance_records, qc_flags


def resolve_concept_label(subdir_name, include_concepts):
    prefix = subdir_name.split("_", 1)[0]
    if prefix.isdigit() and int(prefix) < len(include_concepts):
        return include_concepts[int(prefix)]
    return subdir_name


def extract_eye_frames(input_video_path, eye, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    if eye == "L":
        crop = "crop=iw/2:ih:0:0"
    else:
        crop = "crop=iw/2:ih:iw/2:0"
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_video_path,
        "-vf",
        crop,
        os.path.join(frames_dir, "frame_%06d.png"),
    ]
    g.run_command(command)


def run_sam_matting(
    sam2matting_launcher,
    sam2matting_repo_dir,
    sam_matting_script,
    eye_frames_dir,
    include_concepts,
    eye_output_dir,
):
    if not os.path.isfile(sam_matting_script):
        raise FileNotFoundError(f"SAM matting driver not found: {sam_matting_script}")
    os.makedirs(eye_output_dir, exist_ok=True)
    command = shlex.split(sam2matting_launcher) + [
        sam_matting_script,
        "--sam_repo_dir",
        os.path.abspath(sam2matting_repo_dir),
        "--video_dir",
        os.path.abspath(eye_frames_dir),
        "--output_dir",
        os.path.abspath(eye_output_dir),
        "--languages",
        "|".join(include_concepts),
        "--frame_idx",
        "0",
    ]
    print(
        "Running SAM matting:",
        " ".join(str(part) for part in command),
        flush=True,
    )
    subprocess.run([str(part) for part in command], check=True)


def collect_instance_alpha_sequences(eye_output_dir):
    instances = []
    subdirs = sorted(
        entry
        for entry in glob.glob(os.path.join(eye_output_dir, "*"))
        if os.path.isdir(entry)
    )
    for instance_index, subdir in enumerate(subdirs):
        alpha_paths = sorted(
            glob.glob(os.path.join(subdir, "*.png"))
            + glob.glob(os.path.join(subdir, "*.jpg"))
        )
        if len(alpha_paths) == 0:
            continue
        instances.append(
            {
                "id": instance_index,
                "concept": os.path.basename(subdir),
                "alpha_paths": alpha_paths,
            }
        )
    return instances


def render_and_qc_eye(
    eye_frames_dir,
    instances,
    output_mov_paths,
    fps,
    qc_frame_interval,
    qc_area_jump_threshold,
):
    frame_paths = sorted(glob.glob(os.path.join(eye_frames_dir, "*.png")))
    if len(frame_paths) == 0:
        raise RuntimeError(f"no eye frames in {eye_frames_dir}")
    sample = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
    height, width = sample.shape[:2]

    writers = [
        g.RawAlphaVideoWriter(output_mov_paths[instance_index], width, height, fps)
        for instance_index in range(len(instances))
    ]
    previous_areas = [None] * len(instances)
    previous_present_count = None
    qc_flags = []
    try:
        for frame_index, frame_path in enumerate(frame_paths):
            frame_bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb16 = frame_rgb.astype(np.uint16) * 257
            present_count = 0
            current_areas = []
            for instance_index, instance in enumerate(instances):
                alpha = load_alpha_frame(
                    instance["alpha_paths"], frame_index, width, height
                )
                area = float((alpha > 0.05).mean())
                current_areas.append(area)
                if area > 0.0:
                    present_count += 1
                alpha16 = np.round(alpha * 65535.0).clip(0, 65535).astype(np.uint16)
                rgba = np.dstack([frame_rgb16, alpha16])
                writers[instance_index].write(rgba)

            if frame_index % qc_frame_interval == 0:
                collect_qc_flags(
                    frame_index,
                    instances,
                    current_areas,
                    present_count,
                    previous_areas,
                    previous_present_count,
                    qc_area_jump_threshold,
                    qc_flags,
                )
                previous_areas = current_areas
                previous_present_count = present_count
    finally:
        for writer in writers:
            writer.close()
    return qc_flags


def load_alpha_frame(alpha_paths, frame_index, width, height):
    if frame_index >= len(alpha_paths):
        return np.zeros((height, width), dtype=np.float32)
    alpha = cv2.imread(alpha_paths[frame_index], cv2.IMREAD_UNCHANGED)
    if alpha is None:
        return np.zeros((height, width), dtype=np.float32)
    scale = 65535.0 if alpha.dtype == np.uint16 else 255.0
    if alpha.ndim == 3:
        if alpha.shape[2] == 4:
            alpha = alpha[:, :, 3]
        else:
            alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
    alpha = alpha.astype(np.float32) / scale
    if alpha.shape[:2] != (height, width):
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0.0, 1.0)


def collect_qc_flags(
    frame_index,
    instances,
    current_areas,
    present_count,
    previous_areas,
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
    for instance_index, area in enumerate(current_areas):
        previous_area = previous_areas[instance_index]
        if previous_area is None or previous_area <= 0.0:
            continue
        if abs(area - previous_area) / previous_area > qc_area_jump_threshold:
            qc_flags.append(
                {
                    "frame_index": frame_index,
                    "type": "area_jump",
                    "instance_id": instances[instance_index]["id"],
                    "concept": instances[instance_index]["concept"],
                    "from_area": previous_area,
                    "to_area": area,
                }
            )


def slugify_concept(concept):
    text = re.sub(r"[^a-z0-9]+", "_", concept.strip().lower()).strip("_")
    if text == "":
        text = "instance"
    return text[:40]


def save_sidecar(sidecar_path, sidecar):
    with open(sidecar_path, "w", encoding="utf-8") as file:
        json.dump(sidecar, file, indent=2, default=str)


def write_qc_log(qc_log_path, sidecar):
    lines = [f"QC report for {sidecar['input_video_path']}"]
    for failure in sidecar["failures"]:
        lines.append(f"FAILURE: {failure}")
    for eye, eye_data in sidecar["eyes"].items():
        lines.append(f"eye {eye}: {len(eye_data['instances'])} instances")
        for flag in eye_data["qc_flags"]:
            lines.append(f"  {eye} {flag}")
    with open(qc_log_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    Fire(main)
