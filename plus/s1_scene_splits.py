import os
import json
import contextlib
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import imageio_ffmpeg
import plus.global_params as g
from fire import Fire
from transformers import AutoModel, AutoProcessor


def main(
    input_video_path: str,
    output_dir: str = None,
    model_name: str = "google/siglip2-base-patch16-384",
    sample_fps: float = 15.0,
    input_size: int = 384,
    batch_size: int = 32,
    window_seconds: float = 1.0,
    threshold: float = 0.08,
    prominence: float = 0.025,
    persistence: int = 3,
    cooldown_seconds: float = 3.0,
    refine_seconds: float = 1.0,
    refine_window_seconds: float = 0.20,
    copy_if_no_boundaries: bool = True,
    copy_streams: bool = False,
    crf: int = 18,
    preset: str = "medium",
    overwrite: bool = False,
):
    ignore_start_seconds = cooldown_seconds

    if output_dir is None:
        video_dir = os.path.dirname(os.path.abspath(input_video_path))
        video_stem = os.path.splitext(os.path.basename(input_video_path))[0]
        output_dir = os.path.join(video_dir, "plus")

    video_stem = os.path.splitext(os.path.basename(input_video_path))[0]
    boundaries_json_path = os.path.join(output_dir, f"{video_stem}_boundaries.json")

    if g.should_skip_output(boundaries_json_path, overwrite):
        return
    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    os.makedirs(output_dir, exist_ok=True)
    if overwrite:
        cleanup_output_segments(output_dir, video_stem)

    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")
    try:
        fps, width, height, frame_count, duration = get_video_properties(video)
    finally:
        video.release()

    print(f"Input video size: {width}x{height}", flush=True)
    print(f"Input video FPS: {fps:.3f}", flush=True)
    print(f"Input video duration: {duration:.3f}s", flush=True)

    torch_device = create_device("cuda")
    model, image_mean, image_std = create_siglip_model(model_name, torch_device)
    autocast_dtype = get_autocast_dtype("fp16", torch_device)

    sample_frame_indices = get_sample_frame_indices(fps, frame_count, sample_fps)
    print(
        f"Sampling {len(sample_frame_indices)} frames at {sample_fps:.3f} FPS",
        flush=True,
    )

    sample_times, sampled_frame_indices, embeddings = infer_video_frame_indices(
        input_video_path,
        sample_frame_indices,
        fps,
        model,
        torch_device,
        autocast_dtype,
        image_mean,
        image_std,
        input_size,
        batch_size,
        "sampled",
    )

    coarse_boundaries, scores = detect_coarse_boundaries(
        sample_times,
        sampled_frame_indices,
        embeddings,
        sample_fps,
        window_seconds,
        threshold,
        prominence,
        persistence,
        cooldown_seconds,
        ignore_start_seconds,
    )

    print(f"Detected {len(coarse_boundaries)} coarse boundaries", flush=True)

    boundaries = refine_boundaries(
        input_video_path,
        coarse_boundaries,
        fps,
        frame_count,
        duration,
        model,
        torch_device,
        autocast_dtype,
        image_mean,
        image_std,
        input_size,
        batch_size,
        refine_seconds,
        refine_window_seconds,
        cooldown_seconds,
        ignore_start_seconds,
    )

    print(f"Detected {len(boundaries)} refined boundaries", flush=True)
    for boundary in boundaries:
        print(
            f"==> boundary at {boundary['time']:.3f}s frame {boundary['frame_index']}",
            flush=True,
        )

    split_video(
        input_video_path,
        output_dir,
        video_stem,
        boundaries,
        duration,
        copy_if_no_boundaries,
        copy_streams,
        crf,
        preset,
    )

    save_boundaries_json(
        boundaries_json_path,
        input_video_path,
        fps,
        width,
        height,
        frame_count,
        duration,
        sample_fps,
        input_size,
        batch_size,
        model_name,
        window_seconds,
        threshold,
        prominence,
        persistence,
        cooldown_seconds,
        ignore_start_seconds,
        refine_seconds,
        refine_window_seconds,
        boundaries,
        scores,
    )

    print(f"==> saved scene boundary data: {boundaries_json_path}", flush=True)


def create_device(device):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    return torch_device


def create_siglip_model(model_name, device):
    print(f"Loading SigLIP 2 model: {model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    image_processor = getattr(processor, "image_processor", processor)
    image_mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
    image_std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
    return model, image_mean, image_std


def get_autocast_dtype(precision, device):
    if device.type != "cuda":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("BF16 is not supported on this CUDA device; using FP16", flush=True)
        return torch.float16
    if precision == "fp32":
        return None
    raise ValueError(f"Unsupported precision: {precision}")


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
    duration = frame_count / fps
    return fps, width, height, frame_count, duration


def get_sample_frame_indices(fps, frame_count, sample_fps):
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than 0")
    duration = frame_count / fps
    sample_count = int(np.floor(duration * sample_fps)) + 1
    sample_times = np.arange(sample_count, dtype=np.float64) / sample_fps
    frame_indices = np.rint(sample_times * fps).astype(np.int64)
    frame_indices = frame_indices[frame_indices < frame_count]
    return np.unique(frame_indices)


def infer_video_frame_indices(
    input_video_path,
    frame_indices,
    fps,
    model,
    device,
    autocast_dtype,
    image_mean,
    image_std,
    input_size,
    batch_size,
    progress_name,
):
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if len(frame_indices) == 0:
        return (
            np.empty((0,), dtype=np.float64),
            frame_indices,
            np.empty((0, 0), dtype=np.float32),
        )

    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    batch_frames = []
    read_frame_indices = []
    embedding_batches = []
    selected_frame_indices = []
    target_position = 0
    first_frame_index = int(frame_indices[0])
    last_frame_index = int(frame_indices[-1])
    current_frame_index = first_frame_index
    processed_count = 0

    try:
        video.set(cv2.CAP_PROP_POS_FRAMES, first_frame_index)
        while current_frame_index <= last_frame_index and target_position < len(
            frame_indices
        ):
            success, frame_bgr = video.read()
            if not success:
                break

            target_frame_index = int(frame_indices[target_position])
            if current_frame_index == target_frame_index:
                batch_frames.append(preprocess_frame(frame_bgr, input_size))
                read_frame_indices.append(current_frame_index)
                target_position += 1

                if len(batch_frames) >= batch_size:
                    embedding_batches.append(
                        infer_frame_batch(
                            model,
                            batch_frames,
                            device,
                            autocast_dtype,
                            image_mean,
                            image_std,
                        )
                    )
                    selected_frame_indices.extend(read_frame_indices)
                    processed_count += len(batch_frames)
                    print_progress(progress_name, processed_count, len(frame_indices))
                    batch_frames = []
                    read_frame_indices = []

            current_frame_index += 1
    finally:
        video.release()

    if len(batch_frames) > 0:
        embedding_batches.append(
            infer_frame_batch(
                model,
                batch_frames,
                device,
                autocast_dtype,
                image_mean,
                image_std,
            )
        )
        selected_frame_indices.extend(read_frame_indices)
        processed_count += len(batch_frames)
        print_progress(progress_name, processed_count, len(frame_indices))

    selected_frame_indices = np.asarray(selected_frame_indices, dtype=np.int64)
    sample_times = selected_frame_indices.astype(np.float64) / fps
    if len(embedding_batches) == 0:
        return sample_times, selected_frame_indices, np.empty((0, 0), dtype=np.float32)
    embeddings = np.concatenate(embedding_batches, axis=0)
    return sample_times, selected_frame_indices, embeddings


def print_progress(progress_name, processed_count, total_count):
    if processed_count == total_count or processed_count % 512 == 0:
        print(
            f"==> {progress_name} SigLIP frames {processed_count}/{total_count}",
            flush=True,
        )


def preprocess_frame(frame_bgr, input_size):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(
        frame_rgb, (input_size, input_size), interpolation=cv2.INTER_AREA
    )
    frame_np = frame_rgb.astype(np.float32) / 255.0
    return np.transpose(frame_np, (2, 0, 1))


def infer_frame_batch(
    model, batch_frames, device, autocast_dtype, image_mean, image_std
):
    batch_np = np.stack(batch_frames, axis=0)
    batch = torch.from_numpy(batch_np).to(device, non_blocking=True)
    batch = normalize_siglip_input(batch, image_mean, image_std)

    autocast_context = contextlib.nullcontext()
    if autocast_dtype is not None:
        autocast_context = torch.autocast(device_type=device.type, dtype=autocast_dtype)

    with torch.inference_mode(), autocast_context:
        output = run_siglip_model(model, batch)
        embeddings = extract_embedding_from_output(output)

    embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    return embeddings.cpu().numpy().astype(np.float32)


def normalize_siglip_input(batch, image_mean, image_std):
    mean = torch.tensor(image_mean, dtype=batch.dtype, device=batch.device).reshape(
        1, 3, 1, 1
    )
    stdv = torch.tensor(image_std, dtype=batch.dtype, device=batch.device).reshape(
        1, 3, 1, 1
    )
    return (batch - mean) / stdv


def run_siglip_model(model, batch):
    if hasattr(model, "get_image_features"):
        return model.get_image_features(pixel_values=batch)
    if hasattr(model, "vision_model"):
        output = model.vision_model(pixel_values=batch)
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0]
    return model(pixel_values=batch)


def extract_embedding_from_output(output):
    if isinstance(output, dict):
        embeddings = extract_embedding_from_dict(output)
        if embeddings is not None:
            return embeddings
    if isinstance(output, (list, tuple)):
        embeddings = extract_embedding_from_sequence(output)
        if embeddings is not None:
            return embeddings
    if torch.is_tensor(output):
        embeddings = tensor_to_embedding(output, "")
        if embeddings is not None:
            return embeddings
    raise RuntimeError("Could not extract SigLIP embeddings from model output")


def extract_embedding_from_dict(output):
    embeddings = []
    for key in [
        "image_embeds",
        "image_features",
        "x_norm_clstoken",
        "x_norm_patchtokens",
        "x_prenorm",
        "cls_token",
        "pooler_output",
        "last_hidden_state",
    ]:
        if key not in output:
            continue
        embedding = tensor_to_embedding(output[key], key)
        if embedding is not None:
            embeddings.append(embedding)

    if len(embeddings) > 0:
        return torch.cat(embeddings, dim=1)

    for value in output.values():
        try:
            return extract_embedding_from_output(value)
        except RuntimeError:
            continue
    return None


def extract_embedding_from_sequence(output):
    embeddings = []
    for value in output:
        try:
            embeddings.append(extract_embedding_from_output(value))
        except RuntimeError:
            continue
    if len(embeddings) == 0:
        return None
    return torch.cat(embeddings, dim=1)


def tensor_to_embedding(tensor, key):
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        if "patch" in key:
            return tensor.mean(dim=1)
        return tensor[:, 0]
    if tensor.ndim == 4:
        return tensor.mean(dim=(2, 3))
    return None


def detect_coarse_boundaries(
    sample_times,
    frame_indices,
    embeddings,
    sample_fps,
    window_seconds,
    threshold,
    prominence,
    persistence,
    cooldown_seconds,
    ignore_start_seconds,
):
    if len(embeddings) == 0:
        return [], []

    window_size = max(1, int(round(window_seconds * sample_fps)))
    persistence_count = max(1, int(persistence))
    scores = compute_window_scores(embeddings, window_size)
    mask = compute_score_mask(
        scores, sample_times, threshold, prominence, ignore_start_seconds, window_size
    )
    runs = find_true_runs(mask)

    boundaries = []
    last_boundary_time = -cooldown_seconds
    for start_index, end_index in runs:
        if end_index - start_index < persistence_count:
            continue
        run_scores = scores[start_index:end_index]
        best_index = start_index + int(np.argmax(run_scores))
        boundary_time = float(sample_times[best_index])
        if boundary_time < ignore_start_seconds:
            continue
        if boundary_time < last_boundary_time + cooldown_seconds:
            continue
        boundaries.append(
            {
                "coarse_time": boundary_time,
                "coarse_frame_index": int(frame_indices[best_index]),
                "coarse_score": float(scores[best_index]),
            }
        )
        last_boundary_time = boundary_time

    return boundaries, scores.tolist()


def compute_window_scores(embeddings, window_size):
    scores = np.zeros((len(embeddings),), dtype=np.float32)
    if len(embeddings) < window_size * 2 + 1:
        return scores

    for index in range(window_size, len(embeddings) - window_size):
        before_embedding = embeddings[index - window_size : index].mean(axis=0)
        after_embedding = embeddings[index : index + window_size].mean(axis=0)
        before_embedding = normalize_np_vector(before_embedding)
        after_embedding = normalize_np_vector(after_embedding)
        scores[index] = 1.0 - float(np.dot(before_embedding, after_embedding))
    return scores


def normalize_np_vector(vector):
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return vector
    return vector / norm


def compute_score_mask(
    scores, sample_times, threshold, prominence, ignore_start_seconds, window_size
):
    mask = np.zeros((len(scores),), dtype=np.bool_)
    background_radius = max(window_size * 2, 1)
    exclusion_radius = max(window_size // 2, 1)
    for index, score in enumerate(scores):
        if sample_times[index] < ignore_start_seconds:
            continue
        if score < threshold:
            continue
        start_index = max(0, index - background_radius)
        end_index = min(len(scores), index + background_radius + 1)
        left_background = scores[
            start_index : max(start_index, index - exclusion_radius)
        ]
        right_background = scores[
            min(end_index, index + exclusion_radius + 1) : end_index
        ]
        background_scores = np.concatenate([left_background, right_background])
        if len(background_scores) == 0:
            background_scores = scores[start_index:end_index]
        background = float(np.median(background_scores))
        if score - background < prominence:
            continue
        mask[index] = True
    return mask


def find_true_runs(mask):
    runs = []
    start_index = None
    for index, value in enumerate(mask):
        if value and start_index is None:
            start_index = index
        elif not value and start_index is not None:
            runs.append((start_index, index))
            start_index = None
    if start_index is not None:
        runs.append((start_index, len(mask)))
    return runs


def refine_boundaries(
    input_video_path,
    coarse_boundaries,
    fps,
    frame_count,
    duration,
    model,
    device,
    autocast_dtype,
    image_mean,
    image_std,
    input_size,
    batch_size,
    refine_seconds,
    refine_window_seconds,
    cooldown_seconds,
    ignore_start_seconds,
):
    boundaries = []
    last_boundary_time = -cooldown_seconds
    for coarse_boundary in coarse_boundaries:
        boundary = refine_boundary(
            input_video_path,
            coarse_boundary,
            fps,
            frame_count,
            duration,
            model,
            device,
            autocast_dtype,
            image_mean,
            image_std,
            input_size,
            batch_size,
            refine_seconds,
            refine_window_seconds,
            ignore_start_seconds,
        )
        if boundary["time"] < ignore_start_seconds:
            continue
        if boundary["time"] < last_boundary_time + cooldown_seconds:
            continue
        boundaries.append(boundary)
        last_boundary_time = boundary["time"]
    boundaries.sort(key=lambda boundary: boundary["time"])
    return boundaries


def refine_boundary(
    input_video_path,
    coarse_boundary,
    fps,
    frame_count,
    duration,
    model,
    device,
    autocast_dtype,
    image_mean,
    image_std,
    input_size,
    batch_size,
    refine_seconds,
    refine_window_seconds,
    ignore_start_seconds,
):
    coarse_time = float(coarse_boundary["coarse_time"])
    start_time = max(ignore_start_seconds, coarse_time - refine_seconds)
    end_time = min(duration, coarse_time + refine_seconds)
    start_frame_index = max(0, int(np.floor(start_time * fps)))
    end_frame_index = min(frame_count - 1, int(np.ceil(end_time * fps)))
    frame_indices = np.arange(start_frame_index, end_frame_index + 1, dtype=np.int64)

    if len(frame_indices) < 3:
        return create_refined_boundary(
            coarse_boundary,
            coarse_time,
            int(round(coarse_time * fps)),
            coarse_boundary["coarse_score"],
        )

    frame_times, read_frame_indices, embeddings = infer_video_frame_indices(
        input_video_path,
        frame_indices,
        fps,
        model,
        device,
        autocast_dtype,
        image_mean,
        image_std,
        input_size,
        batch_size,
        "refine",
    )
    if len(embeddings) < 3:
        return create_refined_boundary(
            coarse_boundary,
            coarse_time,
            int(round(coarse_time * fps)),
            coarse_boundary["coarse_score"],
        )

    refine_window_size = max(1, int(round(refine_window_seconds * fps)))
    scores = compute_refine_scores(embeddings, refine_window_size)
    best_index = int(np.argmax(scores))
    refined_time = float(frame_times[best_index])
    refined_frame_index = int(read_frame_indices[best_index])
    refined_score = float(scores[best_index])
    return create_refined_boundary(
        coarse_boundary, refined_time, refined_frame_index, refined_score
    )


def create_refined_boundary(coarse_boundary, time, frame_index, score):
    return {
        "time": float(time),
        "frame_index": int(frame_index),
        "score": float(score),
        "coarse_time": float(coarse_boundary["coarse_time"]),
        "coarse_frame_index": int(coarse_boundary["coarse_frame_index"]),
        "coarse_score": float(coarse_boundary["coarse_score"]),
    }


def compute_refine_scores(embeddings, window_size):
    scores = np.zeros((len(embeddings),), dtype=np.float32)
    for index in range(1, len(embeddings)):
        adjacent_score = 1.0 - float(np.dot(embeddings[index - 1], embeddings[index]))
        scores[index] = max(scores[index], adjacent_score)

    if len(embeddings) < window_size * 2 + 1:
        return scores

    window_scores = compute_window_scores(embeddings, window_size)
    return np.maximum(scores, window_scores)


def split_video(
    input_video_path,
    output_dir,
    video_stem,
    boundaries,
    duration,
    copy_if_no_boundaries,
    copy_streams,
    crf,
    preset,
):
    if len(boundaries) == 0:
        output_path = os.path.join(output_dir, f"{video_stem}_scene_000.mp4")
        if copy_if_no_boundaries:
            if copy_streams:
                copy_video_streams(input_video_path, output_path)
            else:
                split_video_segment_reencode(
                    input_video_path, output_path, 0.0, duration, crf, preset
                )
            print(f"==> saved single scene video: {output_path}", flush=True)
        return

    if copy_streams:
        split_video_copy(input_video_path, output_dir, video_stem, boundaries)
    else:
        split_video_reencode(
            input_video_path, output_dir, video_stem, boundaries, duration, crf, preset
        )


def split_video_copy(input_video_path, output_dir, video_stem, boundaries):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    output_pattern = os.path.join(output_dir, f"{video_stem}_scene_%03d.mp4")
    segment_times = ",".join(f"{boundary['time']:.6f}" for boundary in boundaries)
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_video_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        "-f",
        "segment",
        "-segment_times",
        segment_times,
        "-reset_timestamps",
        "1",
        output_pattern,
    ]
    g.run_command(command)


def copy_video_streams(input_video_path, output_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_video_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        output_path,
    ]
    g.run_command(command)


def split_video_reencode(
    input_video_path, output_dir, video_stem, boundaries, duration, crf, preset
):
    start_times = [0.0] + [float(boundary["time"]) for boundary in boundaries]
    end_times = [float(boundary["time"]) for boundary in boundaries] + [duration]
    for segment_index, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        output_path = os.path.join(
            output_dir, f"{video_stem}_scene_{segment_index:03d}.mp4"
        )
        split_video_segment_reencode(
            input_video_path,
            output_path,
            start_time,
            max(0.001, end_time - start_time),
            crf,
            preset,
        )


def split_video_segment_reencode(
    input_video_path, output_path, start_time, duration, crf, preset
):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_video_path,
        "-ss",
        f"{start_time:.6f}",
        "-t",
        f"{duration:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        output_path,
    ]
    g.run_command(command)


def cleanup_output_segments(output_dir, video_stem):
    prefix = f"{video_stem}_scene_"
    for filename in os.listdir(output_dir):
        if filename.startswith(prefix) and filename.endswith(".mp4"):
            os.remove(os.path.join(output_dir, filename))


def save_boundaries_json(
    boundaries_json_path,
    input_video_path,
    fps,
    width,
    height,
    frame_count,
    duration,
    sample_fps,
    input_size,
    batch_size,
    model_name,
    window_seconds,
    threshold,
    prominence,
    persistence,
    cooldown_seconds,
    ignore_start_seconds,
    refine_seconds,
    refine_window_seconds,
    boundaries,
    scores,
):
    data = {
        "input_video_path": input_video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration": duration,
        "config": {
            "sample_fps": sample_fps,
            "input_size": input_size,
            "batch_size": batch_size,
            "model_name": model_name,
            "window_seconds": window_seconds,
            "threshold": threshold,
            "prominence": prominence,
            "persistence": persistence,
            "cooldown_seconds": cooldown_seconds,
            "ignore_start_seconds": ignore_start_seconds,
            "refine_seconds": refine_seconds,
            "refine_window_seconds": refine_window_seconds,
        },
        "boundaries": boundaries,
        "scores": scores,
    }
    with open(boundaries_json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


if __name__ == "__main__":
    Fire(main)
