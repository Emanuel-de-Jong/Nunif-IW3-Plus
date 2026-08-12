import os
import cv2
import torch
import numpy as np
import plus.global_params as g
from fire import Fire

GREEN = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def main(
    input_video_path: str,
    output_video_path: str = None,
    rvm_model_path: str = "resnet50",
    rvm_downsample_ratio: float = 0.25,
    foreground_bias: float = 0.05,
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

    if rvm_downsample_ratio <= 0 or rvm_downsample_ratio > 1:
        raise ValueError(
            f"rvm_downsample_ratio must be greater than 0 and at most 1, got: {rvm_downsample_ratio}"
        )

    model, device = create_rvm_model(rvm_model_path)
    recurrent_state_left = [None] * 4
    recurrent_state_right = [None] * 4

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

            left_mask, recurrent_state_left = get_foreground_mask(
                model,
                left_frame,
                recurrent_state_left,
                device,
                rvm_downsample_ratio,
            )
            right_mask, recurrent_state_right = get_foreground_mask(
                model,
                right_frame,
                recurrent_state_right,
                device,
                rvm_downsample_ratio,
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


def create_rvm_model(rvm_model_path):
    if rvm_model_path == "resnet50":
        model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            "resnet50",
            trust_repo=True,
        )
    else:
        model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            "resnet50",
            pretrained=False,
            trust_repo=True,
        )
        model.load_state_dict(torch.load(rvm_model_path, map_location="cpu"))

    device = torch.device("cuda")
    return model.to(device).eval(), device


def get_foreground_mask(
    model,
    frame_rgb,
    recurrent_state,
    device,
    downsample_ratio,
):
    frame_tensor = (
        torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )
    with torch.inference_mode():
        _foreground, alpha, *new_state = model(
            frame_tensor,
            *recurrent_state,
            downsample_ratio,
        )
    return alpha[0, 0].cpu().numpy(), new_state


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
