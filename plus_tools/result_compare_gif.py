import sys
import math
import shutil
import random
import subprocess
from pathlib import Path

IMAGEMAGICK = "/home/graviton/base/apps/tools/imagemagick/ImageMagick-7.1.2-24-gcc-x86_64.AppImage"

START_TIME = 3.0
END_TIME = 5.5
DURATION = END_TIME - START_TIME
FPS = 16
CROP = "183:254:351:350"


def main():
    target_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    base_video = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    play_speed = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35

    if play_speed <= 0:
        print("Play speed must be greater than 0")
        sys.exit(1)

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
        frame_file = target_dir / f"frame_{filename}.gif"
        crop_file = target_dir / f"crop_{filename}.gif"

        print(f"\nProcessing: {video_file.name}")

        frame_result = extract_animation(video_file, frame_file, play_speed)
        if frame_result is True:
            frame_count += 1
        elif frame_result is False:
            frame_skipped += 1
        else:
            continue

        crop_result = crop_animation(video_file, crop_file, play_speed)
        if crop_result is True:
            crop_count += 1
        elif crop_result is False:
            crop_skipped += 1

    if base_video is None:
        create_grid(target_dir, play_speed)
    else:
        base_crop_file = target_dir / f"crop_{base_video.stem}.gif"
        create_list(target_dir, base_crop_file, play_speed)

    print(f"\nAnimations created: {frame_count}, existing: {frame_skipped}")
    print(f"Crops created:      {crop_count}, existing: {crop_skipped}")
    print("Done!")


def animation_duration(play_speed):
    return DURATION / play_speed


def run_ffmpeg_gif(inputs, filter_complex, output_file, map_label="out"):
    palette_filter = (
        f"{filter_complex};"
        f"[{map_label}]split[a][b];"
        f"[a]palettegen=stats_mode=diff[p];"
        f"[b][p]paletteuse=dither=sierra2_4a"
    )

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

    for input_args in inputs:
        command.extend(input_args)

    command.extend(
        [
            "-filter_complex",
            palette_filter,
            "-loop",
            "0",
            "-y",
            str(output_file),
        ]
    )

    return subprocess.run(command).returncode == 0


def extract_animation(video_file, frame_file, play_speed):
    if frame_file.exists():
        print(f"Animation exists: {frame_file.name}")
        return False

    success = run_ffmpeg_gif(
        [
            [
                "-ss",
                str(START_TIME),
                "-t",
                str(DURATION),
                "-i",
                str(video_file),
            ]
        ],
        (f"[0:v]" f"setpts=(PTS-STARTPTS)/{play_speed}," f"fps={FPS}" f"[out]"),
        frame_file,
    )

    if success:
        print(f"Created animation: {frame_file.name}")
        return True

    print(f"Failed to extract animation: {video_file.name}")
    frame_file.unlink(missing_ok=True)
    return None


def crop_animation(video_file, crop_file, play_speed):
    if crop_file.exists():
        print(f"Crop exists: {crop_file.name}")
        return False

    success = run_ffmpeg_gif(
        [
            [
                "-ss",
                str(START_TIME),
                "-t",
                str(DURATION),
                "-i",
                str(video_file),
            ]
        ],
        (
            f"[0:v]"
            f"crop={CROP},"
            f"setpts=(PTS-STARTPTS)/{play_speed},"
            f"fps={FPS}"
            f"[out]"
        ),
        crop_file,
    )

    if success:
        print(f"Created crop: {crop_file.name}")
        return True

    print(f"Failed to crop: {video_file.name}")
    crop_file.unlink(missing_ok=True)
    return None


def normalize_filter(index, width=180, height=245):
    return (
        f"[{index}:v]"
        f"fps={FPS},"
        f"scale={width}:{height}:force_original_aspect_ratio=disable,"
        f"setpts=PTS-STARTPTS"
    )


def create_grid(target_dir, play_speed):
    crop_files = sorted(target_dir.glob("crop_*.gif"), key=lambda p: p.name)

    if not crop_files:
        print("\nNo cropped animations found. Skipping grid.")
        return

    duration = animation_duration(play_speed)
    total = len(crop_files)
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)
    output_file = target_dir / "compare_grid.gif"

    print(f"\nCreating animated grid: {cols} columns x {rows} rows")

    inputs = []
    filters = []

    for i, crop_file in enumerate(crop_files):
        inputs.append(
            [
                "-stream_loop",
                "-1",
                "-t",
                str(duration),
                "-i",
                str(crop_file),
            ]
        )
        filters.append(f"{normalize_filter(i)}[v{i}]")

    next_index = total
    row_labels = []

    for row in range(rows):
        row_inputs = []

        for col in range(cols):
            index = row * cols + col

            if index < total:
                row_inputs.append(f"[v{index}]")
            else:
                filters.append(
                    f"color=c=black:s=180x245:r={FPS}:d={duration}[blank{next_index}]"
                )
                row_inputs.append(f"[blank{next_index}]")
                next_index += 1

        row_label = f"row{row}"
        filters.append(
            "".join(row_inputs) + f"hstack=inputs={cols}:shortest=1[{row_label}]"
        )
        row_labels.append(f"[{row_label}]")

    if rows == 1:
        filters.append(f"{row_labels[0]}null[out]")
    else:
        filters.append("".join(row_labels) + f"vstack=inputs={rows}:shortest=1[out]")

    success = run_ffmpeg_gif(
        inputs,
        ";".join(filters),
        output_file,
    )

    if success:
        print(f"Created animated grid: {output_file.name}")
    else:
        print("Failed to create animated grid")
        output_file.unlink(missing_ok=True)


def escape_drawtext(text):
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(",", "\\,")
    )


def create_compare_row(
    base_crop_file,
    left_crop,
    right_crop,
    output_file,
    duration,
):
    left_label = escape_drawtext(left_crop.stem.removeprefix("crop_"))
    right_label = (
        escape_drawtext(right_crop.stem.removeprefix("crop_"))
        if right_crop is not None
        else ""
    )

    inputs = [
        [
            "-stream_loop",
            "-1",
            "-t",
            str(duration),
            "-i",
            str(base_crop_file),
        ],
        [
            "-stream_loop",
            "-1",
            "-t",
            str(duration),
            "-i",
            str(left_crop),
        ],
    ]

    filters = [
        f"{normalize_filter(0)}[base1]",
        f"{normalize_filter(1)}[left]",
    ]

    if right_crop is not None:
        inputs.append(
            [
                "-stream_loop",
                "-1",
                "-t",
                str(duration),
                "-i",
                str(right_crop),
            ]
        )
        filters.append(f"{normalize_filter(2)}[right]")
    else:
        filters.append(f"color=c=black:s=180x245:r={FPS}:d={duration}[right]")

    filters.extend(
        [
            "[base1]split=2[baseleft][baseright]",
            "[baseleft][left][right][baseright]" "hstack=inputs=4:shortest=1[row]",
            (
                "[row]"
                "drawtext="
                "fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
                f"text='{left_label}':"
                "fontsize=10:"
                "fontcolor=white:"
                "borderw=1:"
                "bordercolor=black:"
                "x=184:"
                "y=2,"
                "drawtext="
                "fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
                f"text='{right_label}':"
                "fontsize=10:"
                "fontcolor=white:"
                "borderw=1:"
                "bordercolor=black:"
                "x=536-text_w:"
                "y=15"
                "[out]"
            ),
        ]
    )

    success = run_ffmpeg_gif(
        inputs,
        ";".join(filters),
        output_file,
    )

    if not success:
        output_file.unlink(missing_ok=True)

    return success


def create_list(target_dir, base_crop_file, play_speed):
    if not base_crop_file.exists():
        print(f"\nBase crop does not exist: {base_crop_file.name}")
        return

    duration = animation_duration(play_speed)

    crop_files = sorted(
        [
            crop_file
            for crop_file in target_dir.glob("crop_*.gif")
            if crop_file.resolve() != base_crop_file.resolve()
        ],
        key=lambda p: p.name,
    )

    if not crop_files:
        print("\nNo comparison crops found. Skipping list.")
        return

    random.shuffle(crop_files)

    row_dir = target_dir / ".compare_rows"
    row_dir.mkdir(exist_ok=True)

    row_files = []

    for i in range(0, len(crop_files), 2):
        left_crop = crop_files[i]
        right_crop = crop_files[i + 1] if i + 1 < len(crop_files) else None
        row_file = row_dir / f"row_{i // 2:04d}.gif"

        if create_compare_row(
            base_crop_file,
            left_crop,
            right_crop,
            row_file,
            duration,
        ):
            row_files.append(row_file)
        else:
            print(f"Failed to create comparison row: {i // 2 + 1}")

    if not row_files:
        print("\nNo comparison rows created. Skipping list.")
        shutil.rmtree(row_dir)
        return

    output_file = target_dir / "compare_list.gif"

    print(f"\nCreating animated list: " f"4 columns x {len(row_files)} rows")

    inputs = []
    filters = []

    for i, row_file in enumerate(row_files):
        inputs.append(
            [
                "-stream_loop",
                "-1",
                "-t",
                str(duration),
                "-i",
                str(row_file),
            ]
        )
        filters.append(f"[{i}:v]" f"fps={FPS}," f"setpts=PTS-STARTPTS" f"[row{i}]")

    if len(row_files) == 1:
        filters.append("[row0]null[out]")
    else:
        filters.append(
            "".join(f"[row{i}]" for i in range(len(row_files)))
            + f"vstack=inputs={len(row_files)}:"
            "shortest=1[out]"
        )

    success = run_ffmpeg_gif(
        inputs,
        ";".join(filters),
        output_file,
    )

    if success:
        print(f"Created animated list: {output_file.name}")
    else:
        print("Failed to create animated list")
        output_file.unlink(missing_ok=True)

    shutil.rmtree(row_dir)


if __name__ == "__main__":
    main()
