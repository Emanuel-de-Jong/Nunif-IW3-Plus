import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import plus.global_params as g
from fire import Fire
from iw3.video_depth_anything_streaming_model import VideoDepthAnythingStreamingModel

GREEN = np.array([0.0, 1.0, 0.0], dtype=np.float32)

MASK_EDGE_SOFTNESS = 0.05


def main(
    input_video_path: str,
    output_video_path: str = None,
    depth_model_type: str = "VDA_Stream_S",
    mask_blur_radius: int = 7,
    foreground_bias: float = 0.05,
    threshold_ema_decay: float = 0.9,
    crf: int = 16,
    preset: str = "slow",
    overwrite: bool = False,
):
    if output_video_path is None:
        video_dir = os.path.dirname(os.path.abspath(input_video_path))
        video_stem = os.path.splitext(os.path.basename(input_video_path))[0]
        output_video_path = os.path.join(
            video_dir, "plus", f"{video_stem}_greenscreen.mp4"
        )

    if g.should_skip_output(output_video_path, overwrite):
        return

    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    left_model = create_depth_model(depth_model_type)
    right_model = create_depth_model(depth_model_type)
    device = left_model.device

    threshold_ema_left = None
    threshold_ema_right = None

    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")
    fps, width, height = get_video_properties(video)
    if width % 2 != 0:
        raise ValueError(f"SBS video width must be even, got: {width}")

    eye_width = width // 2
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    output_writer = g.RawVideoWriter(
        output_video_path,
        width,
        height,
        fps,
        codec="libx264",
        crf=crf,
        preset=preset,
        pixel_format="yuv420p",
    )

    frame_index = 0
    try:
        while True:
            success, frame_bgr = video.read()
            if not success:
                break

            frame_rgb = (
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            left_frame = frame_rgb[:, :eye_width]
            right_frame = frame_rgb[:, eye_width:]

            left_depth_np = infer_and_normalize_depth(
                left_model, left_frame, device, height, eye_width
            )
            right_depth_np = infer_and_normalize_depth(
                right_model, right_frame, device, height, eye_width
            )

            threshold_ema_left = update_threshold_ema(
                left_depth_np, threshold_ema_left, threshold_ema_decay
            )
            threshold_ema_right = update_threshold_ema(
                right_depth_np, threshold_ema_right, threshold_ema_decay
            )

            left_mask = compute_foreground_mask(
                left_depth_np, threshold_ema_left, mask_blur_radius
            )
            right_mask = compute_foreground_mask(
                right_depth_np, threshold_ema_right, mask_blur_radius
            )

            left_output = composite_green(left_frame, left_mask, foreground_bias)
            right_output = composite_green(right_frame, right_mask, foreground_bias)
            output_writer.write(np.concatenate([left_output, right_output], axis=1))

            frame_index += 1
            if frame_index % 25 == 0:
                print(f"==> green-screened {frame_index} frames", flush=True)
    finally:
        video.release()
        output_writer.close()

    print(f"==> saved green-screen video: {output_video_path}", flush=True)


def create_depth_model(depth_model_type):
    model = VideoDepthAnythingStreamingModel(depth_model_type)
    model.load(gpu=0)
    return model


def infer_and_normalize_depth(
    model, frame_rgb_hwc, device, target_height, target_width
):
    frame_tensor = torch.from_numpy(frame_rgb_hwc).permute(2, 0, 1).float().to(device)
    with torch.inference_mode():
        depth = model.infer(frame_tensor)

    depth_up = (
        F.interpolate(
            depth.unsqueeze(0),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )
    depth_np = depth_up.cpu().numpy()

    depth_min = depth_np.min()
    depth_max = depth_np.max()
    depth_range = depth_max - depth_min
    if depth_range > 0:
        return (depth_np - depth_min) / depth_range
    return np.zeros_like(depth_np)


def update_threshold_ema(depth_np, current_ema, ema_decay):
    depth_uint8 = (depth_np * 255).clip(0, 255).astype(np.uint8)
    threshold_val, _ = cv2.threshold(
        depth_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    raw_threshold = threshold_val / 255.0
    if current_ema is None:
        return raw_threshold
    return ema_decay * current_ema + (1.0 - ema_decay) * raw_threshold


def compute_foreground_mask(depth_np, threshold, mask_blur_radius):
    depth_uint8 = (depth_np * 255).clip(0, 255).astype(np.uint8)
    depth_smoothed = (
        cv2.bilateralFilter(depth_uint8, d=9, sigmaColor=40, sigmaSpace=40).astype(
            np.float32
        )
        / 255.0
    )

    alpha = 1.0 / (1.0 + np.exp(-(depth_smoothed - threshold) / MASK_EDGE_SOFTNESS))

    if mask_blur_radius > 0:
        blur_kernel_size = 2 * mask_blur_radius + 1
        alpha = cv2.GaussianBlur(alpha, (blur_kernel_size, blur_kernel_size), 0)

    return alpha.clip(0.0, 1.0).astype(np.float32)


def get_video_properties(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        raise ValueError("Could not read video FPS")
    if width <= 0 or height <= 0:
        raise ValueError("Could not read video size")
    return fps, width, height


def composite_green(frame_rgb, mask, foreground_bias):
    alpha = np.clip(mask + foreground_bias, 0.0, 1.0).astype(np.float32)[:, :, None]
    return frame_rgb * alpha + GREEN.reshape(1, 1, 3) * (1.0 - alpha)


if __name__ == "__main__":
    Fire(main)
