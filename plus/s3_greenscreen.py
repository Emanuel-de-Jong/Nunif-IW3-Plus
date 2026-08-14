import os
import gc
import re
import json
import shutil
import tempfile
import cv2
import torch
import numpy as np
import imageio_ffmpeg
import plus.global_params as g
from fire import Fire
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


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
    sam_repo_dir: str = None,
    sam_checkpoint_path: str = None,
    sam_bpe_path: str = None,
    sam_prompt_frame_idx: int = 5,
    sam_prompt_groups: int = 6,
    sam_max_long_side: int = 0,
    sam_video_crf: int = 12,
    sam_video_preset: str = "veryfast",
    sam_compile: bool = False,
    output_alpha_video: bool = True,
    output_instance_videos: bool = False,
    sam_mask_close_kernel: int = 9,
    sam_mask_dilate_kernel: int = 3,
    sam_mask_border_shift: int = 0,
    sam_mask_overlap_gap_fill: int = 50,
    qc_frame_interval: int = 15,
    qc_area_jump_threshold: float = 0.40,
    greenscreen_crf: int = 18,
    greenscreen_preset: str = "medium",
    depth_foreground_threshold: float = 0.2,
    depth_mask_border_shift: int = -10,
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

    if g.should_skip_output(greenscreen_path, overwrite):
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

    qwen_prompts_path = g.PLUS_DIR / "greenscreen_prompts.json"
    if not qwen_prompts_path.exists():
        print("greenscreen_prompts.json couldn't be found!.")
        print(
            "Make a copy of plus/greenscreen_prompts_example.json and name it greenscreen_prompts.json."
        )
        print("Exiting...")
        return

    with open(qwen_prompts_path, "r", encoding="utf-8") as qwen_prompts_file:
        qwen_prompts = json.load(qwen_prompts_file)

    eye_width = width // 2
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, f".{output_stem}_work")
    os.makedirs(work_dir, exist_ok=True)

    config = {
        "vlm_model_id": vlm_model_id,
        "num_sampled_frames": num_sampled_frames,
        "num_vote_runs": num_vote_runs,
        "vlm_temperature": vlm_temperature,
        "vlm_max_new_tokens": vlm_max_new_tokens,
        "vlm_max_long_side": vlm_max_long_side,
        "sam_model_id": sam_model_id,
        "sam_repo_dir": sam_repo_dir,
        "sam_checkpoint_path": sam_checkpoint_path,
        "sam_bpe_path": sam_bpe_path,
        "sam_prompt_frame_idx": sam_prompt_frame_idx,
        "sam_prompt_groups": sam_prompt_groups,
        "sam_max_long_side": sam_max_long_side,
        "sam_video_crf": sam_video_crf,
        "sam_video_preset": sam_video_preset,
        "sam_compile": sam_compile,
        "output_alpha_video": output_alpha_video,
        "output_instance_videos": output_instance_videos,
        "sam_mask_close_kernel": sam_mask_close_kernel,
        "sam_mask_dilate_kernel": sam_mask_dilate_kernel,
        "sam_mask_border_shift": sam_mask_border_shift,
        "sam_mask_overlap_gap_fill": sam_mask_overlap_gap_fill,
        "qc_frame_interval": qc_frame_interval,
        "qc_area_jump_threshold": qc_area_jump_threshold,
        "greenscreen_crf": greenscreen_crf,
        "greenscreen_preset": greenscreen_preset,
        "depth_foreground_threshold": depth_foreground_threshold,
        "depth_mask_border_shift": depth_mask_border_shift,
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
        "specific_include": [],
        "exclude": [],
        "sam_prompts": [],
        "eyes": {},
        "outputs": {},
        "failures": [],
    }

    try:
        sampled_frame_indices = get_sampled_frame_indices(
            frame_count, num_sampled_frames
        )
        sidecar["sampled_frame_indices"] = [
            int(index) for index in sampled_frame_indices
        ]
        frames_dir = os.path.join(work_dir, "samples")
        image_paths = extract_sample_frames(
            input_video_path,
            sampled_frame_indices,
            eye_width,
            frames_dir,
            vlm_max_long_side,
        )
        if len(image_paths) == 0:
            sidecar["failures"].append("frame sampling produced no frames")
            print("==> frame sampling produced no frames", flush=True)
            save_sidecar(sidecar_path, sidecar)
            write_qc_log(qc_log_path, sidecar)
            return

        print(f"==> sampled {len(image_paths)} frames for VLM", flush=True)
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

        try:
            for eye in ("L", "R"):
                crop_eye_video(
                    input_video_path,
                    eye,
                    eye_video_paths[eye],
                    sam_max_long_side,
                    sam_video_crf,
                    sam_video_preset,
                )

            eye_depth_masks = {}
            if depth_foreground_threshold > 0:
                depth_model = create_depth_model(device)
                try:
                    for eye in ("L", "R"):
                        print(f"==> computing depth masks for eye {eye}", flush=True)
                        eye_depth_masks[eye] = compute_eye_depth_masks(
                            depth_model,
                            eye_video_paths[eye],
                            depth_foreground_threshold,
                        )
                finally:
                    del depth_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            predictor = create_sam_predictor(
                sam_model_id,
                sam_repo_dir,
                sam_checkpoint_path,
                sam_bpe_path,
                sam_compile,
                qwen_prompts,
            )
            try:
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    for eye in ("L", "R"):
                        eye_data = process_eye_with_sam(
                            predictor,
                            eye_video_paths[eye],
                            eye_green_paths[eye],
                            eye_alpha_paths[eye],
                            output_dir,
                            output_stem,
                            eye,
                            prompts,
                            sam_prompt_frame_idx,
                            fps,
                            output_alpha_video,
                            output_instance_videos,
                            sam_mask_close_kernel,
                            sam_mask_dilate_kernel,
                            sam_mask_border_shift,
                            sam_mask_overlap_gap_fill,
                            qc_frame_interval,
                            qc_area_jump_threshold,
                            greenscreen_crf,
                            greenscreen_preset,
                            eye_depth_masks.get(eye),
                            depth_foreground_threshold,
                            depth_mask_border_shift,
                        )
                        sidecar["eyes"][eye] = eye_data
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

    broad_prompts = [
        "all foreground subjects and attached objects",
        "complete foreground subject including clothing hair accessories and held objects",
    ]
    for prompt in reversed(broad_prompts):
        key = normalize_concept(prompt)
        if key not in seen:
            prompts.insert(0, prompt)
            seen.add(key)

    if sam_prompt_groups > 0:
        prompts = prompts[:sam_prompt_groups]
    return prompts


def prompt_matches_scene(concept, include_concepts, specific_include_concepts):
    text = normalize_concept(" ".join(include_concepts + specific_include_concepts))
    concept_key = normalize_concept(concept)
    if concept_key in (
        "foreground subject",
        "all foreground object",
        "held object",
        "object in hand",
    ):
        return True
    for word in concept_key.split(" "):
        if len(word) >= 4 and word in text:
            return True
    return False


def create_depth_model(device):
    from iw3.zoedepth_model import ZoeDepthModel

    print("Loading ZoeD_Any_N depth model", flush=True)
    depth_model = ZoeDepthModel("ZoeD_Any_N")
    depth_model.load(gpu=0)
    return depth_model


def compute_eye_depth_masks(depth_model, eye_video_path, depth_foreground_threshold):
    video_frames, width, height, _ = load_video_frames(eye_video_path)
    depth_masks = []
    for frame_rgb in video_frames:
        frame_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(
            2, 0, 1
        )
        depth = depth_model.infer(frame_tensor.to(depth_model.device))
        depth = depth_model.minmax_normalize_chw(depth)
        depth_np = depth.squeeze(0).cpu().numpy()
        mask = (depth_np >= 1.0 - depth_foreground_threshold).astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        depth_masks.append(mask)
    return depth_masks


def apply_depth_mask(
    alpha,
    depth_masks,
    frame_index,
    height,
    width,
    depth_foreground_threshold,
    depth_mask_border_shift=0,
):
    if (
        depth_foreground_threshold <= 0
        or depth_masks is None
        or frame_index >= len(depth_masks)
    ):
        return alpha
    mask = depth_masks[frame_index]
    if depth_mask_border_shift != 0:
        shift = abs(int(depth_mask_border_shift))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (shift * 2 + 1, shift * 2 + 1)
        )
        if depth_mask_border_shift > 0:
            mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            mask = cv2.erode(mask, kernel, iterations=1)
    return np.maximum(alpha, mask.astype(np.float32))


def create_sam_predictor(
    sam_model_id,
    sam_repo_dir,
    checkpoint_path,
    bpe_path,
    compiled,
    qwen_prompts,
):
    print(f"Loading SAM 3 model: {sam_model_id}", flush=True)
    from transformers import Sam3VideoModel, Sam3VideoProcessor

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
    eye_video_path,
    green_video_path,
    alpha_video_path,
    output_dir,
    video_stem,
    eye,
    prompts,
    prompt_frame_idx,
    fps,
    output_alpha_video,
    output_instance_videos,
    sam_mask_close_kernel,
    sam_mask_dilate_kernel,
    sam_mask_border_shift,
    sam_mask_overlap_gap_fill,
    qc_frame_interval,
    qc_area_jump_threshold,
    greenscreen_crf=18,
    greenscreen_preset="veryfast",
    depth_masks=None,
    depth_foreground_threshold=0.0,
    depth_mask_border_shift=0,
):
    video_frames, width, height, frame_count = load_video_frames(eye_video_path)
    model = predictor["model"]
    processor = predictor["processor"]
    device = predictor["device"]
    dtype = predictor["dtype"]

    try:
        inference_session = processor.init_video_session(
            video=video_frames,
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=dtype,
        )
        prompt_frame_idx = min(max(int(prompt_frame_idx), 0), frame_count - 1)
        for prompt in prompts:
            inference_session = processor.add_text_prompt(
                inference_session=inference_session,
                text=prompt,
            )

        green_writer = g.RawVideoWriter(
            green_video_path,
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
                alpha_video_path,
                width,
                height,
                fps,
                codec="libx264",
                crf=12,
                preset="veryfast",
                pixel_format="yuv420p",
            )

        instance_writers = {}
        instance_records = {}
        previous_area = None
        previous_present_count = None
        max_instances = 0
        qc_flags = []
        current_frame_index = 0

        try:
            for model_outputs in model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=frame_count - 1,
            ):
                frame_index = int(model_outputs.frame_idx)
                while current_frame_index < frame_index:
                    frame_rgb = video_frames[current_frame_index]
                    empty_alpha = apply_depth_mask(
                        np.zeros((height, width), dtype=np.float32),
                        depth_masks,
                        current_frame_index,
                        height,
                        width,
                        depth_foreground_threshold,
                        depth_mask_border_shift,
                    )
                    green_writer.write(composite_green(frame_rgb, empty_alpha))
                    if alpha_writer is not None:
                        alpha_u8 = (
                            np.round(empty_alpha * 255.0).clip(0, 255).astype(np.uint8)
                        )
                        alpha_writer.write(cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2RGB))
                    current_frame_index += 1

                processed_outputs = processor.postprocess_outputs(
                    inference_session,
                    model_outputs,
                )
                masks = tensor_to_numpy(processed_outputs.get("masks", None))
                object_ids = tensor_to_list(processed_outputs.get("object_ids", []))
                combined_alpha, present_count = combine_masks(
                    masks,
                    height,
                    width,
                    sam_mask_close_kernel,
                    sam_mask_dilate_kernel,
                    sam_mask_border_shift,
                    sam_mask_overlap_gap_fill,
                )
                combined_alpha = apply_depth_mask(
                    combined_alpha,
                    depth_masks,
                    frame_index,
                    height,
                    width,
                    depth_foreground_threshold,
                    depth_mask_border_shift,
                )
                max_instances = max(max_instances, present_count)
                frame_rgb = video_frames[frame_index]
                green_frame = composite_green(frame_rgb, combined_alpha)
                green_writer.write(green_frame)
                if alpha_writer is not None:
                    alpha_u8 = (
                        np.round(combined_alpha * 255.0).clip(0, 255).astype(np.uint8)
                    )
                    alpha_writer.write(cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2RGB))
                if output_instance_videos and masks is not None:
                    write_instance_frames(
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
                        qc_flags,
                    )
                    previous_area = area
                    previous_present_count = present_count
                current_frame_index += 1

            while current_frame_index < frame_count:
                frame_rgb = video_frames[current_frame_index]
                empty_alpha = apply_depth_mask(
                    np.zeros((height, width), dtype=np.float32),
                    depth_masks,
                    current_frame_index,
                    height,
                    width,
                    depth_foreground_threshold,
                    depth_mask_border_shift,
                )
                green_writer.write(composite_green(frame_rgb, empty_alpha))
                if alpha_writer is not None:
                    alpha_u8 = (
                        np.round(empty_alpha * 255.0).clip(0, 255).astype(np.uint8)
                    )
                    alpha_writer.write(cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2RGB))
                current_frame_index += 1
        finally:
            green_writer.close()
            if alpha_writer is not None:
                alpha_writer.close()
            for writer in instance_writers.values():
                writer.close()

        return {
            "max_instances": max_instances,
            "instances": list(instance_records.values()),
            "qc_flags": qc_flags,
        }
    finally:
        del video_frames


def load_video_frames(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open cropped eye video: {video_path}")
    try:
        _, width, height, frame_count = get_video_properties(video)
        frames = []
        while True:
            success, frame_bgr = video.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        video.release()
    if len(frames) == 0:
        raise ValueError(f"Could not read cropped eye video frames: {video_path}")
    if len(frames) != frame_count:
        frame_count = len(frames)
    return frames, width, height, frame_count


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
    sam_mask_dilate_kernel,
    sam_mask_border_shift,
    sam_mask_overlap_gap_fill,
):
    if masks is None:
        return np.zeros((height, width), dtype=np.float32), 0
    masks = np.asarray(masks)
    if masks.size == 0:
        return np.zeros((height, width), dtype=np.float32), 0
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

    combined = fill_overlap_gaps(combined, instance_masks, sam_mask_overlap_gap_fill)
    combined = postprocess_mask(
        combined, sam_mask_close_kernel, sam_mask_dilate_kernel, sam_mask_border_shift
    )
    return combined.astype(np.float32) / 255.0, present_count


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


def postprocess_mask(
    mask_u8, sam_mask_close_kernel, sam_mask_dilate_kernel, sam_mask_border_shift
):
    if sam_mask_close_kernel > 1:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (sam_mask_close_kernel, sam_mask_close_kernel)
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)
    if sam_mask_dilate_kernel > 1:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (sam_mask_dilate_kernel, sam_mask_dilate_kernel)
        )
        mask_u8 = cv2.dilate(mask_u8, dilate_kernel, iterations=1)
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
