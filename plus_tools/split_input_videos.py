import os
import cv2
from fire import Fire
from pathlib import Path

MIN_DURATION_SECONDS = 5
MAX_DURATION_SECONDS = 120 + 5

MIN_RESOLUTION = 700
# Compare to the videos biggest side (width or height)
MAX_RESOLUTION = 3840

MIN_FPS = 24 - 3
# Look at the average fps, not the peak
MAX_FPS = 60 + 5


def main(path: str):
    video_dir_path = Path(path)
    include_dir_path = video_dir_path / "include"
    exclude_dir_path = video_dir_path / "exclude"

    print("Starting splitting...\n")

    video_paths = []
    for video_path in video_dir_path.iterdir():
        if not video_path.is_file():
            continue

        video_paths.append(video_path)

    total_videos = len(video_paths)

    for idx, video_path in enumerate(video_paths):
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            video.release()
            continue

        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0

        video.release()

        is_duration_in_range = MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS
        is_res_in_range = (
            min(width, height) >= MIN_RESOLUTION
            and max(width, height) <= MAX_RESOLUTION
        )
        is_fps_in_range = MIN_FPS <= fps <= MAX_FPS

        should_include = is_duration_in_range and is_res_in_range and is_fps_in_range

        destination = include_dir_path if should_include else exclude_dir_path
        include_dir_path.mkdir(exist_ok=True)
        exclude_dir_path.mkdir(exist_ok=True)

        os.replace(video_path, destination / video_path.name)

        status_str = "include"
        if not should_include:
            status_str = "exclude-"
            if not is_duration_in_range:
                status_str += "duration"
            elif not is_res_in_range:
                status_str += "resolution"
            elif not is_fps_in_range:
                status_str += "fps"

        print(f"({idx + 1}/{total_videos}) [{status_str}] {video_path.name}")

    print("\nSplitting done!")


if __name__ == "__main__":
    Fire(main)
