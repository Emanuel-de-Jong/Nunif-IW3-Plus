import os
import re
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CHECKPOINT = "checkpoints/SAM2Matting-SAM3.pt"
BPE_PATH = "sam3/bpe_simple_vocab_16e6.txt.gz"
DEVICE = "cuda"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam_repo_dir", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--languages", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()

    sam_repo_dir = os.path.abspath(args.sam_repo_dir)
    video_dir = os.path.abspath(args.video_dir)
    output_dir = os.path.abspath(args.output_dir)
    sys.path.insert(0, sam_repo_dir)
    os.chdir(sam_repo_dir)

    predictor = build_language_predictor(CHECKPOINT, compiled=args.compiled)

    concepts = [
        concept for concept in args.languages.split("|") if concept.strip() != ""
    ]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for concept_index, concept in enumerate(concepts):
            concept_output_dir = os.path.join(
                output_dir, f"{concept_index:02d}_{slugify_concept(concept)}"
            )
            print(f"==> matting concept {concept_index}: {concept}", flush=True)
            process_language_to_png(
                video_dir, concept_output_dir, predictor, concept, args.frame_idx
            )


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


def process_language_to_png(video_dir, output_dir, predictor, language, frame_idx):
    frame_files = sorted(
        [
            f
            for f in os.listdir(video_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    if not frame_files:
        raise RuntimeError(f"no frames in {video_dir}")
    os.makedirs(output_dir, exist_ok=True)
    sample = Image.open(os.path.join(video_dir, frame_files[0]))
    sample_width, sample_height = sample.size

    resp = predictor.handle_request(
        dict(
            type="start_session",
            resource_path=video_dir,
        )
    )
    session_id = resp["session_id"]

    predictor.model.add_prompt(
        inference_state=predictor._get_session(session_id)["state"],
        frame_idx=frame_idx,
        text_str=language,
    )

    seen = set()
    for resp in predictor.handle_stream_request(
        dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction="forward",
        )
    ):
        idx = resp["frame_index"]
        if idx in seen:
            continue
        seen.add(idx)
        state = predictor._get_session(session_id)["state"]
        alpha_2d = compute_alpha_or_zero(
            predictor.model, state, idx, resp["outputs"], sample_height, sample_width
        )
        write_alpha_png(output_dir, frame_files[idx], alpha_2d)

    for frame_position, frame_name in enumerate(frame_files):
        if frame_position not in seen:
            zeros = np.zeros((sample_height, sample_width), dtype=np.float32)
            write_alpha_png(output_dir, frame_name, zeros)

    predictor.handle_request(dict(type="close_session", session_id=session_id))


def compute_alpha_or_zero(
    model, state, frame_idx, outputs, fallback_height, fallback_width
):
    masks = outputs.get("out_binary_masks")
    if masks is None or len(masks) == 0:
        return np.zeros((fallback_height, fallback_width), dtype=np.float32)
    mask = pick_mask(outputs)
    return compute_alpha(model, state, frame_idx, mask.cpu().numpy()).clip(0, 1)


def pick_mask(outputs):
    masks = outputs["out_binary_masks"]
    if len(masks) == 0:
        raise RuntimeError("no mask from propagate")
    if torch.is_tensor(masks):
        mask = masks[0]
    else:
        mask = torch.as_tensor(masks[0])
    if mask.dim() == 3:
        mask = mask[0]
    return mask


def ensure_frame_cache(model, state, frame_idx):
    if frame_idx not in state["feature_cache"]:
        model._prepare_backbone_feats(state, frame_idx, reverse=False)


def compute_alpha(model, state, frame_idx, mask_hw):
    ensure_frame_cache(model, state, frame_idx)
    image, cache = state["feature_cache"][frame_idx]
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(DEVICE)
    fpn = cache["tracker_backbone_out"]["backbone_fpn"]
    high_res_features = list(fpn)
    mask = torch.as_tensor(mask_hw > 0, device=DEVICE, dtype=torch.float32)
    mask_288 = F.interpolate(
        mask[None, None],
        size=(288, 288),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    binary_mask_288 = (mask_288 > 0.0).float()
    alpha, _, _ = model.tracker._forward_alpha_heads(
        input=image,
        backbone_features=None,
        point_inputs=None,
        mask_inputs=binary_mask_288,
        unknown_region_inputs=None,
        high_res_features=high_res_features,
        image=None,
        trimap_input=None,
    )
    video_h = state["orig_height"]
    video_w = state["orig_width"]
    alpha_up = F.interpolate(
        alpha.float(),
        size=(video_h, video_w),
        mode="bilinear",
        align_corners=False,
    )
    return alpha_up.squeeze().detach().float().cpu().numpy()


def write_alpha_png(output_dir, frame_name, alpha_2d):
    stem = os.path.splitext(frame_name)[0]
    alpha_uint16 = np.round(np.clip(alpha_2d, 0.0, 1.0) * 65535.0).astype(np.uint16)
    cv2.imwrite(os.path.join(output_dir, f"{stem}.png"), alpha_uint16)


def slugify_concept(concept):
    text = re.sub(r"[^a-z0-9]+", "_", concept.strip().lower()).strip("_")
    if text == "":
        text = "instance"
    return text[:40]


if __name__ == "__main__":
    main()
