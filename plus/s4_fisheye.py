import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
import plus.global_params as g
from fire import Fire

GREEN_BGR = (0, 255, 0)
# GREEN_BGR = (64, 230, 43)


def main(
    input_video_path: str,
    output_video_path: str = None,
    output_dir: str = None,
    fisheye_input_video_path: str = None,
    source_hfov: float = 80.0,
    fit_mode: str = "cover",
    fill: float = 0.75,
    mapping_fov: float = 180.0,
    min_mapping_fov: float = 60.0,
    max_mapping_fov: float = 100.0,
    scale: float = 1.3,
    crf: int = 18,
    preset: str = "medium",
    overwrite: bool = False,
):
    video_dir = os.path.dirname(os.path.abspath(input_video_path))
    video_stem = os.path.splitext(os.path.basename(input_video_path))[0]

    if output_dir is None:
        output_dir = os.path.join(video_dir, "plus", "tmp")

    if fisheye_input_video_path is None:
        fisheye_input_video_path = os.path.join(
            output_dir,
            f"{video_stem}_3_green.mp4",
        )

    if output_video_path is None:
        output_video_path = os.path.join(
            output_dir,
            f"{video_stem}_4_fish.mp4",
        )

    if g.should_skip_output(output_video_path, overwrite):
        return

    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    if not os.path.isfile(fisheye_input_video_path):
        raise FileNotFoundError(
            f"Fisheye input video not found: {fisheye_input_video_path}"
        )

    if not 1.0 < source_hfov < 179.0:
        raise ValueError("source_hfov must be between 1 and 179 degrees")

    if not 0 < scale <= 2:
        raise ValueError("scale must be between 0 and 2")

    if not 0 < fill <= 1:
        raise ValueError("fill must be between 0 and 1")

    if fit_mode not in ("cover", "literal"):
        raise ValueError('fit_mode must be "cover" or "literal"')

    if mapping_fov is not None:
        if not 1.0 < mapping_fov <= 180.0:
            raise ValueError("mapping_fov must be between 1 and 180 degrees")

    os.makedirs(
        os.path.dirname(output_video_path) or ".",
        exist_ok=True,
    )

    width, height, fps, total_frames = probe(fisheye_input_video_path)

    if width % 2:
        raise ValueError("Input width must be divisible by 2 for full-SBS video")

    eye_w = width // 2
    eye_h = height

    out_eye = eye_w
    out_w = out_eye * 2
    out_h = out_eye

    source_vfov = calculate_vfov(
        source_hfov,
        eye_w,
        eye_h,
    )

    effective_mapping_fov = choose_mapping_fov(
        source_hfov=source_hfov,
        source_vfov=source_vfov,
        fit_mode=fit_mode,
        fill=fill,
        explicit_mapping_fov=mapping_fov,
        min_mapping_fov=min_mapping_fov,
        max_mapping_fov=max_mapping_fov,
    )

    map_x, map_y = make_map(
        src_w=eye_w,
        src_h=eye_h,
        out_size=out_eye,
        source_hfov=source_hfov,
        mapping_fov=effective_mapping_fov,
        scale=scale,
    )

    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            fisheye_input_video_path,
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
    )

    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{out_w}x{out_h}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "pipe:0",
            "-i",
            fisheye_input_video_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx265",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "hvc1",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_primaries",
            "bt709",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-shortest",
            output_video_path,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
    )

    frame_bytes = width * height * 3
    frame_number = 0

    print()
    print(f"Input:               {width}x{height}")
    print(f"Per eye:             {eye_w}x{eye_h}")
    print(f"Output:              {out_w}x{out_h}")
    print()
    print(f"Assumed source HFOV: {source_hfov:.2f}°")
    print(f"Derived source VFOV: {source_vfov:.2f}°")
    print(f"Fit mode:            {fit_mode}")
    print(f"Mapping FOV:         {effective_mapping_fov:.2f}°")
    print(f"Circle scale:        {scale:.3f}")

    if fit_mode == "cover" and mapping_fov is None:
        print(f"Target fill:         {fill:.2f}")

    print(f"Circle diameter:     " f"{round(out_eye * scale)} px")

    horizontal_fill = min(
        1.0,
        source_hfov / effective_mapping_fov,
    )

    vertical_fill = min(
        1.0,
        source_vfov / effective_mapping_fov,
    )

    print(f"Approx H fill:       " f"{horizontal_fill * 100:.1f}% radius")

    print(f"Approx V fill:       " f"{vertical_fill * 100:.1f}% radius")

    print()

    try:
        while True:
            raw = decoder.stdout.read(frame_bytes)

            if not raw:
                break

            if len(raw) != frame_bytes:
                print("\nWarning: incomplete final frame received")
                break

            frame = np.frombuffer(
                raw,
                dtype=np.uint8,
            ).reshape(
                height,
                width,
                3,
            )

            left = frame[:, :eye_w]
            right = frame[:, eye_w:]

            left = convert_eye(
                left,
                map_x,
                map_y,
            )

            right = convert_eye(
                right,
                map_x,
                map_y,
            )

            output = np.hstack((left, right))

            try:
                encoder.stdin.write(output.tobytes())
            except BrokenPipeError:
                break

            frame_number += 1

            if frame_number % 10 == 0:
                if total_frames:
                    pct = frame_number / total_frames * 100

                    print(
                        f"\r{frame_number}/{total_frames} " f"({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r{frame_number}",
                        end="",
                        flush=True,
                    )

    finally:
        if decoder.stdout:
            decoder.stdout.close()

        if encoder.stdin:
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass

    decoder.wait()
    encoder.wait()

    print()

    if decoder.returncode != 0:
        print(decoder.stderr.read().decode(errors="replace"))
        sys.exit(1)

    if encoder.returncode != 0:
        print(encoder.stderr.read().decode(errors="replace"))
        sys.exit(1)

    print(f"Finished: {output_video_path}")


def parse_fraction(value):
    if "/" in value:
        a, b = value.split("/")
        b = float(b)

        if b == 0:
            raise ValueError(f"Invalid fraction: {value}")

        return float(a) / b

    return float(value)


def probe(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            ("stream=" "width," "height," "avg_frame_rate," "nb_frames," "duration"),
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    streams = json.loads(result.stdout).get("streams", [])

    if not streams:
        raise RuntimeError(f"No video stream found in {path}")

    stream = streams[0]

    width = int(stream["width"])
    height = int(stream["height"])

    fps = parse_fraction(stream["avg_frame_rate"])

    frames = stream.get("nb_frames")

    frames = int(frames) if frames and frames != "N/A" else None

    if frames is None:
        duration = stream.get("duration")

        if duration and duration != "N/A":
            frames = round(float(duration) * fps)

    return width, height, fps, frames


def calculate_vfov(
    hfov_degrees,
    width,
    height,
):
    hfov = math.radians(hfov_degrees)

    vfov = 2.0 * math.atan((height / width) * math.tan(hfov / 2.0))

    return math.degrees(vfov)


def choose_mapping_fov(
    source_hfov,
    source_vfov,
    fit_mode,
    fill,
    explicit_mapping_fov,
    min_mapping_fov,
    max_mapping_fov,
):
    if explicit_mapping_fov is not None:
        return explicit_mapping_fov

    if fit_mode == "literal":
        return 180.0

    narrow_fov = min(
        source_hfov,
        source_vfov,
    )

    calculated = narrow_fov / fill

    calculated = max(
        min_mapping_fov,
        calculated,
    )

    calculated = min(
        max_mapping_fov,
        calculated,
    )

    return calculated


def make_map(
    src_w,
    src_h,
    out_size,
    source_hfov,
    mapping_fov,
    scale,
):
    source_hfov_rad = math.radians(source_hfov)

    mapping_half_fov = math.radians(mapping_fov / 2.0)

    fx = src_w / (2.0 * math.tan(source_hfov_rad / 2.0))

    fy = fx

    src_cx = (src_w - 1) / 2.0

    src_cy = (src_h - 1) / 2.0

    center = (out_size - 1) / 2.0

    radius = out_size * scale / 2.0

    x = np.arange(
        out_size,
        dtype=np.float32,
    )

    y = np.arange(
        out_size,
        dtype=np.float32,
    )

    xx, yy = np.meshgrid(
        x,
        y,
    )

    dx = xx - center
    dy = yy - center

    r = np.sqrt(dx * dx + dy * dy)

    nr = r / radius

    theta = nr * mapping_half_fov

    phi = np.arctan2(
        dy,
        dx,
    )

    sin_theta = np.sin(theta)

    rx = sin_theta * np.cos(phi)

    ry = sin_theta * np.sin(phi)

    rz = np.cos(theta)

    safe_z = np.where(
        rz > 1e-8,
        rz,
        1e-8,
    )

    map_x = src_cx + fx * rx / safe_z

    map_y = src_cy + fy * ry / safe_z

    valid = (
        (nr <= 1.0)
        & (rz > 0.0)
        & (map_x >= 0.0)
        & (map_x <= src_w - 1)
        & (map_y >= 0.0)
        & (map_y <= src_h - 1)
    )

    map_x = map_x.astype(np.float32)

    map_y = map_y.astype(np.float32)

    map_x[~valid] = -1.0
    map_y[~valid] = -1.0

    return map_x, map_y


def convert_eye(
    eye,
    map_x,
    map_y,
):
    return cv2.remap(
        eye,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=GREEN_BGR,
    )


if __name__ == "__main__":
    Fire(main)
