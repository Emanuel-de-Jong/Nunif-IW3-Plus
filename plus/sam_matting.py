import os
import re
import sys
import gc
import json
import shutil
import argparse
import cv2
import numpy as np
import torch
from PIL import Image

CHECKPOINT = "checkpoints/SAM2Matting-SAM3.pt"
BPE_PATH = "sam3/bpe_simple_vocab_16e6.txt.gz"
DEVICE = "cuda"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam_repo_dir", type=str, required=True)
    parser.add_argument("--video_dirs", type=str, required=True)
    parser.add_argument("--output_dirs", type=str, required=True)
    parser.add_argument("--languages", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()

    sam_repo_dir = os.path.abspath(args.sam_repo_dir)
    video_dirs = [os.path.abspath(part) for part in args.video_dirs.split("|") if part]
    output_dirs = [
        os.path.abspath(part) for part in args.output_dirs.split("|") if part
    ]
    concepts = [
        concept for concept in args.languages.split("|") if concept.strip() != ""
    ]
    sys.path.insert(0, sam_repo_dir)
    os.chdir(sam_repo_dir)

    detector = build_language_predictor(CHECKPOINT, compiled=args.compiled)
    eye_detections = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for video_dir in video_dirs:
            detection_dir = video_dir + "_det"
            make_detection_dir(video_dir, args.frame_idx, detection_dir)
            detections = detect_frame0_instances(detector, detection_dir, concepts)
            print(
                f"==> detected {len(detections)} instances in {video_dir}", flush=True
            )
            eye_detections.append(detections)
            shutil.rmtree(detection_dir, ignore_errors=True)
    del detector
    gc.collect()
    torch.cuda.empty_cache()

    tracker = build_tracker_predictor(CHECKPOINT, compiled=args.compiled)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for video_dir, output_dir, detections in zip(
            video_dirs, output_dirs, eye_detections
        ):
            print(
                f"==> tracking {len(detections)} instances in {video_dir}", flush=True
            )
            track_and_write(tracker, video_dir, output_dir, detections, args.frame_idx)
    del tracker
    gc.collect()
    torch.cuda.empty_cache()


def build_language_predictor(checkpoint, compiled=False):
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(
        gpus_to_use=[0],
        checkpoint_path=checkpoint,
        strict_state_dict_loading=False,
        bpe_path=BPE_PATH,
    )
    if compiled:
        trunk = predictor.model.detector.backbone.vision_backbone.trunk
        trunk.forward = torch.compile(
            trunk.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
        from sam3.model.utils.trt import replace_unknown_alpha_predictor_with_trt

        predictor.model.tracker = replace_unknown_alpha_predictor_with_trt(
            predictor.model.tracker
        )
    return predictor


def load_tracker_state_dict(checkpoint):
    from iopath.common.file_io import g_pathmgr

    with g_pathmgr.open(checkpoint, "rb") as file:
        ckpt = torch.load(file, map_location="cpu", weights_only=True)
    state_dict = ckpt["model"]
    out = {}
    for key, value in state_dict.items():
        if key.startswith("detector.backbone.vision_backbone."):
            out[key.removeprefix("detector.")] = value
        elif key.startswith("tracker."):
            out[key.removeprefix("tracker.")] = value
    return out


def build_tracker_predictor(checkpoint, compiled=False):
    from sam3.model.sam3matting_video_predictor import build_sam3matting_video_predictor

    state_dict = load_tracker_state_dict(checkpoint)
    predictor = build_sam3matting_video_predictor(checkpoint=None, device=DEVICE)
    missing, unexpected = predictor.load_state_dict(state_dict, strict=False)
    print("missing keys: ", missing)
    print("unexpected keys: ", unexpected)
    if compiled:
        trunk = predictor.backbone.vision_backbone.trunk
        trunk.forward = torch.compile(
            trunk.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
        from sam3.model.utils.trt import replace_unknown_alpha_predictor_with_trt

        predictor = replace_unknown_alpha_predictor_with_trt(predictor)
    return predictor


def make_detection_dir(video_dir, frame_idx, detection_dir):
    frame_files = sorted_frame_files(video_dir)
    if len(frame_files) == 0:
        raise RuntimeError(f"no frames in {video_dir}")
    if os.path.isdir(detection_dir):
        shutil.rmtree(detection_dir, ignore_errors=True)
    os.makedirs(detection_dir, exist_ok=True)
    source_name = frame_files[min(frame_idx, len(frame_files) - 1)]
    shutil.copy2(
        os.path.join(video_dir, source_name),
        os.path.join(detection_dir, source_name),
    )


def detect_frame0_instances(detector, detection_dir, concepts):
    detections = []
    for concept in concepts:
        resp = detector.handle_request(
            dict(type="start_session", resource_path=detection_dir)
        )
        session_id = resp["session_id"]
        detector.model.add_prompt(
            inference_state=detector._get_session(session_id)["state"],
            frame_idx=0,
            text_str=concept,
        )
        masks = None
        for resp in detector.handle_stream_request(
            dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",
            )
        ):
            masks = resp["outputs"].get("out_binary_masks")
            break
        detector.handle_request(dict(type="close_session", session_id=session_id))
        if masks is None:
            continue
        for mask in masks:
            mask_2d = to_2d_numpy(mask)
            if mask_2d is None or float((mask_2d > 0).mean()) <= 0.0:
                continue
            detections.append({"concept": concept, "mask": mask_2d})
    return detections


def to_2d_numpy(mask):
    if torch.is_tensor(mask):
        array = mask.detach().float().cpu().numpy()
    else:
        array = np.asarray(mask, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim != 2:
        return None
    return array


def track_and_write(tracker, video_dir, output_dir, detections, frame_idx):
    frame_files = sorted_frame_files(video_dir)
    if len(frame_files) == 0:
        raise RuntimeError(f"no frames in {video_dir}")
    os.makedirs(output_dir, exist_ok=True)
    sample = Image.open(os.path.join(video_dir, frame_files[0]))
    sample_width, sample_height = sample.size

    instance_dirs = []
    manifest = []
    for index, detection in enumerate(detections):
        instance_dir = os.path.join(output_dir, f"{index:03d}")
        os.makedirs(instance_dir, exist_ok=True)
        instance_dirs.append(instance_dir)
        manifest.append({"index": index, "concept": detection["concept"]})
    with open(
        os.path.join(output_dir, "instances.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)
    if len(detections) == 0:
        return

    state = tracker.init_state(video_path=video_dir)
    tracker.reset_state(state)
    for index, detection in enumerate(detections):
        mask_tensor = torch.from_numpy(detection["mask"]).to(DEVICE)
        tracker.add_new_mask(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=index + 1,
            mask=mask_tensor,
        )

    seen = set()
    for (
        out_frame_idx,
        _obj_ids,
        _masks,
        alpha_np,
        _unknown,
    ) in tracker.propagate_in_video(state):
        seen.add(out_frame_idx)
        frame_name = frame_files[out_frame_idx]
        for position in range(len(instance_dirs)):
            write_alpha_png(instance_dirs[position], frame_name, alpha_np[position])

    for frame_position, frame_name in enumerate(frame_files):
        if frame_position not in seen:
            zeros = np.zeros((sample_height, sample_width), dtype=np.float32)
            for instance_dir in instance_dirs:
                write_alpha_png(instance_dir, frame_name, zeros)


def sorted_frame_files(video_dir):
    return sorted(
        entry
        for entry in os.listdir(video_dir)
        if entry.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def write_alpha_png(output_dir, frame_name, alpha_2d):
    stem = os.path.splitext(frame_name)[0]
    alpha = np.asarray(alpha_2d, dtype=np.float32)
    alpha_uint16 = np.round(np.clip(alpha, 0.0, 1.0) * 65535.0).astype(np.uint16)
    cv2.imwrite(os.path.join(output_dir, f"{stem}.png"), alpha_uint16)


def slugify_concept(concept):
    text = re.sub(r"[^a-z0-9]+", "_", concept.strip().lower()).strip("_")
    if text == "":
        text = "instance"
    return text[:40]


if __name__ == "__main__":
    main()
