## Nunif IW3 Plus

Nunif IW3 Plus is a personal fork of nunif focused on iw3 video conversion.

It adds optional post-processing for IW3 side-by-side videos. The default Plus pipeline replaces the background with green and then remaps the video to a fisheye-style view. Upscaling and scene splitting helpers are also included, but are not enabled in the default `run_plus.sh` pipeline.

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

#### About NUNIF_HOME

If the environment variable `NUNIF_HOME` is defined, downloaded pretrained models, configuration files, cache, temporary files, and lock files will be saved under `NUNIF_HOME`. This may be useful when packaging or in situations where the source directory does not have write permissions.
The `~` character at the beginning of a path string is expanded to the home directory.

### License Notes

Note that if you distribute binary builds, it is possible that it will be GPL.

This is due to PyAV(av) wheel package containing the GPL version of ffmpeg library.
You can build PyAV with the LGPL version of ffmpeg library.

If you load this repository with torch.hub.load for waifu2x Python API etc, this problem does not exist because PyAV is not a dependent package.
