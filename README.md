## Nunif IW3 Plus

Adds upscaling and greenscreening.

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
- [PyTorch](https://pytorch.org/get-started/locally/)
- See requirements.txt

We usually support the latest version. If there are bugs or compatibility issues, we will specify the version.

- [INSTALL-ubuntu](INSTALL-ubuntu.md)
- [INSTALL-windows](INSTALL-windows.md)
- [INSTALL-macos](INSTALL-macos.md)

For Intel GPUs, additionally see section [INSTALL-xpu](INSTALL-xpu.md).  
For older NVIDIA GPUs, additionally see section [INSTALL-cu126](INSTALL-cu126.md).


For container, packages, or special hardware builds, see [extra_build](extra_build).

#### About NUNIF_HOME

If the environment variable `NUNIF_HOME` is defined, downloaded pretrained models, configuration files, cache, temporary files, and lock files will be saved under `NUNIF_HOME`. This may be useful when packaging or in situations where the source directory does not have write permissions.
The `~` character at the beginning of a path string is expanded to the home directory.

### License Notes

Note that if you distribute binary builds, it is possible that it will be GPL.

This is due to PyAV(av) wheel package containing the GPL version of ffmpeg library.
You can build PyAV with the LGPL version of ffmpeg library.

If you load this repository with torch.hub.load for waifu2x Python API etc, this problem does not exist because PyAV is not a dependent package.
