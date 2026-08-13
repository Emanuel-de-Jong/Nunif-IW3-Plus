import sys
import math
import shutil
import subprocess
from pathlib import Path

IMAGEMAGICK = "/home/graviton/base/apps/tools/imagemagick/ImageMagick-7.1.2-24-gcc-x86_64.AppImage"


def main():
    target_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    base_video = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if base_video is not None and not base_video.is_absolute():
        base_video = target_dir / base_video

    frame_count = 0
    crop_count = 0
    frame_skipped = 0
    crop_skipped = 0

    video_files = sorted(target_dir.glob("*.mp4"))

    if base_video is not None:
        if not base_video.exists():
            print(f"Base compare video does not exist: {base_video}")
            sys.exit(1)

        if base_video not in video_files:
            video_files.append(base_video)

    for video_file in video_files:
        filename = video_file.stem
        frame_file = target_dir / f"frame_{filename}.png"
        crop_file = target_dir / f"crop_{filename}.png"

        print(f"\nProcessing: {video_file.name}")

        frame_result = extract_frame(video_file, frame_file)
        if frame_result is True:
            frame_count += 1
        elif frame_result is False:
            frame_skipped += 1
        else:
            continue

        crop_result = crop_frame(video_file, frame_file, crop_file)
        if crop_result is True:
            crop_count += 1
        elif crop_result is False:
            crop_skipped += 1

    if base_video is None:
        create_grid(target_dir)
    else:
        base_crop_file = target_dir / f"crop_{base_video.stem}.png"
        create_list(target_dir, base_crop_file)

    print(f"\nFrames created: {frame_count}, existing: {frame_skipped}")
    print(f"Crops created:  {crop_count}, existing: {crop_skipped}")
    print("Done!")


def extract_frame(video_file, frame_file):
    if frame_file.exists():
        print(f"Frame exists: {frame_file.name}")
        return False

    result = subprocess.run(
        [
            "ffmpeg",
            "-ss",
            "00:00:03",
            "-i",
            str(video_file),
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(frame_file),
            "-y",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode == 0:
        print(f"Created frame: {frame_file.name}")
        return True

    print(f"Failed to extract frame: {video_file.name}")
    return None


def crop_frame(video_file, frame_file, crop_file):
    if crop_file.exists():
        print(f"Crop exists: {crop_file.name}")
        return False

    result = subprocess.run(
        [
            IMAGEMAGICK,
            str(frame_file),
            "-crop",
            "245x245+352+360",
            "+repage",
            str(crop_file),
        ]
    )

    if result.returncode == 0:
        print(f"Created crop: {crop_file.name}")
        return True

    print(f"Failed to crop: {video_file.name}")
    crop_file.unlink(missing_ok=True)
    return None


def create_grid(target_dir):
    crop_files = sorted(target_dir.glob("crop_*.png"), key=lambda p: p.name)

    if not crop_files:
        print("\nNo cropped images found. Skipping grid.")
        return

    total = len(crop_files)
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)
    output_file = target_dir / "compare_grid.png"

    print(f"\nCreating grid: {cols} columns x {rows} rows")

    result = subprocess.run(
        [
            IMAGEMAGICK,
            "montage",
            *map(str, crop_files),
            "-tile",
            f"{cols}x{rows}",
            "-geometry",
            "+0+0",
            str(output_file),
        ]
    )

    if result.returncode == 0:
        print(f"Created grid: {output_file.name}")
    else:
        print("Failed to create grid")


def create_compare_row(base_crop_file, left_crop, right_crop, output_file):
    left_label = left_crop.stem.removeprefix("crop_")
    right_label = (
        right_crop.stem.removeprefix("crop_") if right_crop is not None else ""
    )

    images = [
        base_crop_file,
        left_crop,
        right_crop if right_crop is not None else base_crop_file,
        base_crop_file,
    ]

    result = subprocess.run(
        [
            IMAGEMAGICK,
            "montage",
            *map(str, images),
            "-tile",
            "4x1",
            "-geometry",
            "+0+0",
            "png:-",
        ],
        stdout=subprocess.PIPE,
    )

    if result.returncode != 0:
        return False

    annotate = subprocess.run(
        [
            IMAGEMAGICK,
            "png:-",
            "-font",
            "DejaVu-Sans",
            "-pointsize",
            "12",
            "-fill",
            "white",
            "-stroke",
            "black",
            "-strokewidth",
            "1",
            "-gravity",
            "NorthWest",
            "-annotate",
            "+4+4",
            left_label,
            "-gravity",
            "NorthEast",
            "-annotate",
            "+4+4",
            right_label,
            str(output_file),
        ],
        input=result.stdout,
    )

    return annotate.returncode == 0


def create_list(target_dir, base_crop_file):
    if not base_crop_file.exists():
        print(f"\nBase crop does not exist: {base_crop_file.name}")
        return

    crop_files = sorted(
        [
            crop_file
            for crop_file in target_dir.glob("crop_*.png")
            if crop_file.resolve() != base_crop_file.resolve()
        ],
        key=lambda p: p.name,
    )

    if not crop_files:
        print("\nNo comparison crops found. Skipping list.")
        return

    row_dir = target_dir / ".compare_rows"
    row_dir.mkdir(exist_ok=True)

    row_files = []

    for i in range(0, len(crop_files), 2):
        left_crop = crop_files[i]
        right_crop = crop_files[i + 1] if i + 1 < len(crop_files) else None
        row_file = row_dir / f"row_{i // 2:04d}.png"

        if create_compare_row(base_crop_file, left_crop, right_crop, row_file):
            row_files.append(row_file)
        else:
            print(f"Failed to create comparison row: {i // 2 + 1}")

    if not row_files:
        print("\nNo comparison rows created. Skipping list.")
        return

    output_file = target_dir / "compare_list.png"

    print(f"\nCreating list: 4 columns x {len(row_files)} rows")

    result = subprocess.run(
        [
            IMAGEMAGICK,
            "montage",
            *map(str, row_files),
            "-tile",
            f"1x{len(row_files)}",
            "-geometry",
            "+0+0",
            str(output_file),
        ]
    )

    if result.returncode == 0:
        print(f"Created list: {output_file.name}")
    else:
        print("Failed to create list")

    shutil.rmtree(row_dir)


if __name__ == "__main__":
    main()
