import cv2
import torch
import numpy as np
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# YOLO
def setup_yolo(weights_path="yolo11n.pt"):
    """Load YOLO model once."""
    yolo_model = YOLO(weights_path)
    class_names = yolo_model.names
    return yolo_model, class_names


# MiDaS
def setup_midas(model_type="MiDaS_small"):
    """Load MiDaS model and transforms once."""
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(DEVICE).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform

def get_depth_map(frame, midas, transform):
    """Compute depth map using MiDaS."""
    input_batch = transform(frame).to(DEVICE)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = (prediction.cpu().numpy() - prediction.min()) / (prediction.max() - prediction.min() + 1e-6)
    return depth


# YOLO + MiDaS
def fuse_yolo_midas(frame, yolo_model, midas, transform, class_id=None):
    """Run YOLO detection + depth estimation. Returns objects, depth_map, frame with boxes."""
    depth_map = get_depth_map(frame, midas, transform)
    results = yolo_model(frame, classes=[class_id] if class_id is not None else None, verbose=False)

    print("\n" + yolo_model.names[class_id] + "\n")

    objects = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = results[0].names[int(box.cls)]
        confidence = float(box.conf)

        # Depth calculation for object
        if x2 > x1 and y2 > y1:
            depth_crop = depth_map[y1:y2, x1:x2]
            median_depth = float(np.median(depth_crop))
        else:
            median_depth = None

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {median_depth:.2f}" if median_depth is not None else label
        cv2.putText(frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Position calculation
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = frame.shape[1], frame.shape[0]
        horizontal = "left" if cx < w/3 else "right" if cx > 2*w/3 else "center"
        vertical   = "top" if cy < h/3 else "bottom" if cy > 2*h/3 else "middle"
        depth_category = None
        if median_depth is not None:
            depth_category = "close" if median_depth > 0.3 else "far"

        objects.append({
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "depth_norm": median_depth,
            "position": vertical + " " + horizontal,
            "depth_category": depth_category
        })

    return objects, depth_map, frame


# convert image to base64
import base64
from io import BytesIO

def to_base64_img(img):
    """Convert OpenCV image to base64 PNG."""
    if len(img.shape) == 2:  # grayscale/depth
        img = (img * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")
