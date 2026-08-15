# SAM 3.1 fixes

SAM 3.1 is pretty much required for `s3_greenscreen.py` because it's 7x faster than SAM 3 and the SAM substep takes unacceptably long otherwise. Problem is, the Hugging Face transformers don't support it, hf hub can't download it and even SAM's official repository has outdated setup steps, an outdated usage example and broken code. This file describes the needed steps to get it working for the 2026-03-26 [model checkpoint](https://huggingface.co/facebook/sam3.1) and the `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da` [repository commit](https://github.com/facebookresearch/sam3/tree/8f0b7f4d4e7eda2ed606ebde6702c93359ad01da).

1. Download `sam3.1_multiplex.pt` from the [hf page](https://huggingface.co/facebook/sam3.1/). You might have to request access first. Put it in `/plus/checkpoints`.
1. In `plus/`, run `git clone https://github.com/facebookresearch/sam3.git`, then `cd sam3` and `git checkout -d 8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`.
1. Copy everything except for this `README.md` from `sam3.1_fixes/` to `sam3/`. Overwrite when prompted.
1. Follow the `## Installation` steps in `README.md`, including step 5. Before step 5, run `pip install "setuptools<81" "scipy>=1.11,<1.14" psutil`.
1. Install the other step 3 dependencies: `pip install imageio imageio-ffmpeg fire transformers qwen-vl-utils accelerate kernels`
1. Optional: To test if it has been installed correctly, change the value of `video_path` in `sam3/test.py` to your own video and run `python test.py`. If a `test.png` got created and it's not completely white or black, it's working!
