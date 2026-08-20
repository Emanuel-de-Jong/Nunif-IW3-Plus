import subprocess
import numpy as np
import imageio_ffmpeg
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = BASE_PATH / "_out"

PLUS_DIR = BASE_PATH / "plus"
VIDEO2X_PATH = PLUS_DIR / "Video2X" / "Video2X-x86_64.AppImage"

CHECKPOINTS_DIR = PLUS_DIR / "checkpoints"
SVD_CHECKPOINTS_PATH = CHECKPOINTS_DIR / "stable-video-diffusion-img2vid-xt-1-1"
DEPTHCRAFTER_UNET_PATH = CHECKPOINTS_DIR / "DepthCrafter"


def should_skip_output(output_path, overwrite=False):
    if Path(output_path).exists() and not overwrite:
        print(f"==> output already exists, skipping: {output_path}", flush=True)
        return True
    return False


def run_command(command):
    print("Running command:", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


class RawVideoWriter:
    def __init__(
        self,
        output_path,
        width,
        height,
        fps,
        codec="libx264",
        crf=12,
        preset="slow",
        pixel_format="yuv420p",
    ):
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            codec,
        ]

        if codec == "ffv1":
            command.extend(
                [
                    "-level",
                    "3",
                    "-coder",
                    "1",
                    "-context",
                    "1",
                    "-g",
                    "1",
                    "-pix_fmt",
                    "yuv444p",
                ]
            )
        else:
            command.extend(
                [
                    "-vf",
                    "crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0",
                    "-crf",
                    str(crf),
                    "-preset",
                    preset,
                    "-pix_fmt",
                    pixel_format,
                ]
            )

        command.append(str(output_path))
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame):
        if self.process.stdin is None:
            raise RuntimeError("Video writer is closed")

        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, self.process.args)


class RawAlphaVideoWriter:
    def __init__(
        self,
        output_path,
        width,
        height,
        fps,
    ):
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgba64le",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
            str(output_path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame):
        if self.process.stdin is None:
            raise RuntimeError("Video writer is closed")

        frame = np.asarray(frame)
        if frame.dtype != np.uint16:
            if frame.max() <= 1.0:
                frame = frame * 65535.0
            frame = np.clip(frame, 0.0, 65535.0).astype(np.uint16)
        frame = np.ascontiguousarray(frame)
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, self.process.args)
