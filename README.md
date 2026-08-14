## Nunif IW3 Plus

Nunif IW3 Plus is a personal fork of nunif focused on iw3 video conversion.

It adds optional post-processing for IW3 side-by-side videos. The default Plus pipeline replaces the background with green (for example for VR passthrough) and then remaps the video to a fisheye-style view. Upscaling and scene splitting helpers are also included, but are not enabled in the default `run_plus.sh` pipeline.

Original repo can be found [here](https://github.com/nagadomi/nunif/tree/dev).

## iw3

[iw3/README.md](./iw3/README.md)

I want to watch any 2D video as 3D video on my VR device, so I developed this very personal tool.

iw3 provides the ability to convert any 2D image/video into side-by-side 3D image/video.

### iw3-desktop

[iw3/docs/desktop.md](./iw3/docs/desktop.md)

iw3.desktop is a tool that converts your PC desktop screen into 3D and streaming over WiFi.

You can watch any image and video/live displayed on your PC as 3D in realtime.

### iw3-player

[iw3/player/README.md](./iw3/player/README.md)

iw3-player is a self-hosted, specialized viewing environment for stereoscopic media.  
It allows you to stream media that has been pre-converted to 3D with iw3 from your PC and enjoy it on VR devices through a WebXR application.

## Install

### Installer for Windows users

- [nunif windows package](windows_package/docs/README.md)
- [nunif windows package (日本語)](windows_package/docs/README_ja.md)

### For developers

#### Dependencies

- Python 3 (developed with 3.12)
- Conda. The included `run.sh`, `run_plus.sh`, and `split_input_videos.sh` scripts activate a conda environment named `nunifiw3`.
- [PyTorch](https://pytorch.org/get-started/locally/)
- See requirements.txt

We usually support the latest version. If there are bugs or compatibility issues, we will specify the version.

Plus processing currently assumes a CUDA-capable NVIDIA GPU. The Plus dependencies are included in `requirements.txt`.

A typical local setup would be:

```
conda create -n nunifiw3 python=3.12
conda activate nunifiw3
pip install -r requirements-torch.txt
pip install -r requirements.txt
pip install -r requirements-gui.txt
```

For greenscreening, request access to [`facebook/sam3`](https://huggingface.co/facebook/sam3) and authenticate Hugging Face locally before running Plus.

Use `hf auth login` or set `HF_TOKEN`. The Qwen3-VL and SAM3 models are downloaded from Hugging Face when first used.

Copy `plus/greenscreen_prompts_example.json` to `plus/greenscreen_prompts.json` before using greenscreening. You can edit that file to change the VLM prompt and common foreground concepts.

`greenscreen_prompts.json` has three keys:

- `system_prompt`: Sets the VLM's role and output format. Keep this short and strict so the response remains valid JSON that the greenscreen step can parse. Only change it if you need to alter the response format or the model's overall behavior.
- `user_instruction`: Describes how the VLM should decide what belongs in the foreground. Use it to tune the compositing rules for your footage, such as whether attached objects, shadows, reflections, or certain scene elements should be included. Be explicit about what to include and exclude, and preserve the requirement to return the expected `include`, `specific_include`, and `exclude` lists.
- `common_concepts`: A list of broad foreground concepts that are always available as segmentation prompts. Add recurring subjects in your videos, such as a particular type of prop, clothing, animal, or vehicle. Prefer simple, recognizable concepts. Remove concepts that repeatedly cause unrelated background objects to be selected.

If you enable the upscaling step in `run_plus.sh`, put the [Video2X AppImage](https://github.com/k4yt3x/video2x/releases/tag/6.4.0) at `plus/Video2X/Video2X-x86_64.AppImage`. The default upscaler uses RealESRGAN 2x and skips upscaling when the result would exceed the configured maximum size.

- [INSTALL-ubuntu](INSTALL-ubuntu.md)
- [INSTALL-windows](INSTALL-windows.md)
- [INSTALL-macos](INSTALL-macos.md)

For Intel GPUs, additionally see section [INSTALL-xpu](INSTALL-xpu.md).  
For older NVIDIA GPUs, additionally see section [INSTALL-cu126](INSTALL-cu126.md).


For container, packages, or special hardware builds, see [extra_build](extra_build).

## Plus usage

Start the IW3 GUI with the conda-based wrapper:

```
./run.sh
```

In the IW3 GUI, the `Enable Plus` checkbox runs `run_plus.sh` automatically after a video conversion finishes. For CLI conversion, use `--plus` to do the same.

You can also run Plus directly on an existing SBS video:

```
./run_plus.sh path/to/video_LRF_Full_SBS.mp4
```

Process all videos in a directory that do not already have a result:

```
./run_plus_batch.sh path/to/directory
```

Or run a step directly:

```
conda activate nunifiw3
python -m plus.s1_scene_splits --input_video_path "PATH_TO_YOUR_VIDEO" --output_dir "SAVE_DIRECTORY_PATH"
```

Plus writes results under a `plus` subdirectory next to the input video, with temporary files in `plus/tmp`. Important per-step parameters can be changed by editing the `python -m` statements in `run_plus.sh`.

### Plus step arguments

All Plus steps support `--input_video_path`. Most also support `--output_video_path` and/or `--output_dir`. These control where the step reads from and writes to, and are already wired up in `run_plus.sh`.

The following tables explain the most important arguments outside of the three mentioned above.

#### Step 1: scene splits (`plus.s1_scene_splits`)

This step detects large visual changes and splits the input video into separate shot files. It is useful when a long video has cuts that confuse later foreground detection.

| Argument | Default | What it does |
|----------|---------|--------------|
| `--cooldown_seconds` | `3.0` | Minimum time between detected cuts. Also avoids detecting cuts right at the start. |
| `--threshold` | `0.08` | Main sensitivity for detecting a cut. Lower values split more often. Higher values split less often. |
| `--prominence` | `0.025` | Requires a cut candidate to stand out from nearby frames. Higher values ignore weaker transitions. |
| `--persistence` | `3` | Number of consecutive samples that must look like a boundary. Higher values reduce false positives. |
| `--refine_seconds` / `--refine_window_seconds` | `1.0` / `0.20` | Searches around the coarse cut to place the final split closer to the real boundary. |
| `--sample_fps` | `15.0` | How many frames per second are checked. Higher values can catch faster cuts, but are slower. |

#### Step 2: upscale (`plus.s2_upscale`)

This step splits the SBS video into left and right views, upscales each view with Video2X/RealESRGAN, then combines them again.

| Argument | Default | What it does |
|----------|---------|--------------|
| `--realesrgan_model` | `realesr-animevideov3` | RealESRGAN model passed to Video2X. Change this if you want a different Video2X-supported model. |
| `--max_width` / `--max_height` | `10240` / `5120` | Safety limit for the final 2x SBS size. If the result would be larger, the step copies the input instead of upscaling. |

#### Step 3: greenscreen (`plus.s3_greenscreen`)

This step asks Qwen VL what objects are in the foreground, uses SAM3 to track their borders and uses ZoeD_Any_N to track the remaining very near areas. It then writes a green-background SBS video.

| Argument | Default | What it does |
|----------|---------|--------------|
| `--num_sampled_frames` | `7` | Number of frames shown to the VLM. More frames can improve prompt quality when the shot changes, but use more VRAM/time. |
| `--num_vote_runs` | `2` | Runs the VLM multiple times and votes on concepts. Higher values can make prompts more stable, but are slower. |
| `--vlm_temperature` | `0.2` | Randomness for VLM answers. Lower values are more deterministic. Higher values can produce more varied guesses. |
| `--vlm_max_long_side` | `1024` | Maximum size of sampled frames sent to the VLM. Lower values use less VRAM, but may miss small objects. |
| `--sam_prompt_groups` | `6` | Maximum number of prompt groups sent to SAM. Higher values can include more object concepts, but can also add unwanted masks. |
| `--sam_max_long_side` | `0` | Optional size limit for SAM processing. `0` keeps the current size. Lower values can reduce VRAM use. |
| `--sam_mask_close_kernel` | `9` | Fills small holes/gaps in the mask. Higher values make masks smoother but can merge nearby areas. |
| `--sam_mask_dilate_kernel` | `3` | Expands the mask edge. Higher values keep more border pixels around the foreground. |
| `--sam_mask_border_shift` | `0` | Shrinks or expands the final mask edge. Negative values shrink it. Positive values expand it. |
| `--sam_mask_overlap_gap_fill` | `50` | Fills gaps between overlapping/nearby SAM instance masks. Higher values connect masks more aggressively. |
| `--depth_foreground_threshold` / `--depth_mask_border_shift` | `0.2` / `-10` | Uses the IW3 depth layout as an extra foreground hint. The border shift trims or expands that depth mask. |

#### Step 4: fisheye (`plus.s4_fisheye`)

This step remaps each SBS eye into a fisheye-style square view and copies audio from the input when available.

| Argument | Default | What it does |
|----------|---------|--------------|
| `--source_hfov` | `80.0` | Assumed horizontal field of view of the source video, in degrees. This changes how strongly the image is warped. |
| `--scale` | `1.3` | Size of the fisheye circle in the output. Higher values zoom in. Lower values show more of the circle. |
| `--expand` | `True` | Allows the step to increase output resolution so the fisheye circle keeps more detail. |
| `--expand_max_eye_size` | `4320` | Maximum per-eye output size when `--expand` is enabled. |

#### About NUNIF_HOME

If the environment variable `NUNIF_HOME` is defined, downloaded pretrained models, configuration files, cache, temporary files, and lock files will be saved under `NUNIF_HOME`. This may be useful when packaging or in situations where the source directory does not have write permissions.
The `~` character at the beginning of a path string is expanded to the home directory.

### License Notes

Note that if you distribute binary builds, it is possible that it will be GPL.

This is due to PyAV(av) wheel package containing the GPL version of ffmpeg library.
You can build PyAV with the LGPL version of ffmpeg library.

If you load this repository with torch.hub.load for waifu2x Python API etc, this problem does not exist because PyAV is not a dependent package.
