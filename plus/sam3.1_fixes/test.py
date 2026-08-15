import os
import cv2
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_multiplex_video_predictor

predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path="/home/graviton/base/code/repos/Nunif-IW3-Plus/plus/checkpoints/sam3.1_multiplex.pt",
    use_fa3=False,
)


def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")


# "video_path" needs to be either a JPEG folder or a MP4 video file
video_path = "/home/graviton/Downloads/test.mp4"

if isinstance(video_path, str) and video_path.endswith(".mp4"):
    cap = cv2.VideoCapture(video_path)
    IMG_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    IMG_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
else:
    frame_files = [
        os.path.join(video_path, name)
        for name in os.listdir(video_path)
        if name.lower().endswith((".jpg", ".jpeg"))
    ]
    with Image.open(frame_files[0]) as image:
        IMG_WIDTH, IMG_HEIGHT = image.size

response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]


# note: in case you already ran one text prompt and now want to switch to another text prompt
# it's required to reset the session first (otherwise the results would be wrong)
_ = predictor.handle_request(
    request=dict(
        type="reset_session",
        session_id=session_id,
    )
)


prompt_text_str = "person"
frame_idx = 0  # add a text prompt on frame 0
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text=prompt_text_str,
    )
)
out = response["outputs"]


# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(predictor, session_id)


# we pick id 2, which is the dancer in the front
obj_id = 2
response = predictor.handle_request(
    request=dict(
        type="remove_object",
        session_id=session_id,
        obj_id=obj_id,
    )
)

# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(predictor, session_id)


# let's add back the dancer via point prompts.
# we will use a single positive click to add the dancer back.

frame_idx = 0
obj_id = 2
points_abs = np.array(
    [
        [760, 550],  # positive click
    ]
)
# positive clicks have label 1, while negative clicks have label 0
labels = np.array([1])

# convert points and labels to tensors; also convert to relative coordinates
points = abs_to_rel_coords(
    points_abs,
    IMG_WIDTH,
    IMG_HEIGHT,
    coord_type="point",
)

response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        points=points,
        point_labels=labels,
        obj_id=obj_id,
    )
)
out = response["outputs"]

# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(predictor, session_id)

frame_idx = 0
out = outputs_per_frame[frame_idx]

obj_ids = np.asarray(out["out_obj_ids"])
masks = np.asarray(out["out_binary_masks"])

mask = masks[np.where(obj_ids == obj_id)[0][0]]
mask = np.squeeze(mask)
Image.fromarray(mask.astype(np.uint8) * 255).save("test.png")

_ = predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
    )
)
