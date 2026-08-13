import os
import shutil
import cv2
import imageio_ffmpeg
import plus.global_params as g
from fire import Fire


def main(
    input_video_path: str,
    output_video_path: str = None,
    video2x_path: str = str(g.VIDEO2X_PATH),
    max_width: int = 10240,
    max_height: int = 5120,
    realesrgan_model: str = "realesr-animevideov3",
    gpu: int = 0,
    crf: int = 18,
    preset: str = "medium",
    overwrite: bool = False,
):
    if output_video_path is None:
        video_dir = os.path.dirname(os.path.abspath(input_video_path))
        video_stem = os.path.splitext(os.path.basename(input_video_path))[0]
        output_video_path = os.path.join(video_dir, "plus", f"{video_stem}_upscale.mp4")

    if g.should_skip_output(output_video_path, overwrite):
        return
    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    width, height = get_video_size(input_video_path)
    output_dir = os.path.dirname(output_video_path) or "."
    print(f"Input video size: {width}x{height}", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    if width * 2 > max_width or height * 2 > max_height:
        print(
            f"2x output would exceed {max_width}x{max_height}; copying original video",
            flush=True,
        )
        shutil.copy2(input_video_path, output_video_path)
        return

    if not os.path.isfile(video2x_path):
        raise FileNotFoundError(f"Video2X AppImage not found: {video2x_path}")
    if width % 2 != 0:
        raise ValueError(f"Stereo video width must be even, got {width}")

    print(f"RealESRGAN 2x output size: {width * 2}x{height * 2}", flush=True)
    output_name = os.path.splitext(os.path.basename(output_video_path))[0]
    left_input_path = os.path.join(output_dir, f".{output_name}_left_input.mkv")
    right_input_path = os.path.join(output_dir, f".{output_name}_right_input.mkv")
    left_output_path = os.path.join(output_dir, f".{output_name}_left_upscaled.mp4")
    right_output_path = os.path.join(output_dir, f".{output_name}_right_upscaled.mp4")

    try:
        split_stereo_video(input_video_path, left_input_path, right_input_path)
        run_video2x(
            video2x_path,
            left_input_path,
            left_output_path,
            realesrgan_model,
            gpu,
        )
        run_video2x(
            video2x_path,
            right_input_path,
            right_output_path,
            realesrgan_model,
            gpu,
        )
        combine_stereo_video(
            left_output_path, right_output_path, output_video_path, crf, preset
        )
    finally:
        for temp_path in [
            left_input_path,
            right_input_path,
            left_output_path,
            right_output_path,
        ]:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def split_stereo_video(input_video_path, left_output_path, right_output_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_video_path,
        "-filter_complex",
        "[0:v]split=2[left][right];[left]crop=iw/2:ih:0:0[leftout];[right]crop=iw/2:ih:iw/2:0[rightout]",
        "-map",
        "[leftout]",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "yuv444p",
        left_output_path,
        "-map",
        "[rightout]",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "yuv444p",
        right_output_path,
    ]
    g.run_command(command)


def combine_stereo_video(
    left_input_path, right_input_path, output_video_path, crf, preset
):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        left_input_path,
        "-i",
        right_input_path,
        "-filter_complex",
        "[0:v][1:v]hstack=inputs=2[out]",
        "-map",
        "[out]",
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


def get_video_size(input_video_path):
    video = cv2.VideoCapture(str(input_video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read video size: {input_video_path}")
    return width, height


def run_video2x(
    video2x_path,
    input_video_path,
    output_video_path,
    realesrgan_model,
    gpu,
):
    command = [
        video2x_path,
        "-i",
        input_video_path,
        "-o",
        output_video_path,
        "-p",
        "realesrgan",
        "-s",
        "2",
        "--realesrgan-model",
        realesrgan_model,
    ]
    if gpu is not None:
        command.extend(["-d", str(gpu)])
    print("Running Video2X", flush=True)
    g.run_command(command)


if __name__ == "__main__":
    Fire(main)
