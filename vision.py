import cv2
import torch
import numpy as np
from ultralytics import YOLO
import math

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
    input_batch = transform(frame).to(DEVICE)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    prediction_np = prediction.cpu().numpy()
    
    depth = (prediction_np - prediction_np.min()) / (prediction_np.max() - prediction_np.min() + 1e-6)
    return depth


# YOLO + MiDaS
def fuse_yolo_midas(frame, yolo_model, midas, transform, class_id=None):
    """Run YOLO detection + depth estimation. Returns objects, depth_map, frame with boxes."""
    depth_map = get_depth_map(frame, midas, transform)
    
    # Ensure class_id is always a list of integers or None
    if class_id is not None:
        # If it's a single value (string, int, float), wrap it in a list
        if isinstance(class_id, (str, int, float)):
            class_id = [int(class_id)]
        # If it's a list/tuple, convert each element to int
        elif isinstance(class_id, (list, tuple)):
            class_id = [int(c) for c in class_id]
        else:
            raise TypeError(f"Unexpected class_id type: {type(class_id)}")

    results = yolo_model(frame, classes=class_id, verbose=False)

    objects = []
    frame_height, frame_width = frame.shape[:2]
    
    bbox_overlap_check = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = results[0].names[int(box.cls)]
        confidence = float(box.conf)
        bbox = [x1, y1, x2, y2]
        bbox_overlap_check.append([x1, y1, x2, y2])

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
        
        # Calculate horizontal position
        horizontal = "center"
        if cx < frame_width / 2:
            horizontal = "left"
        elif cx > frame_width / 2:
            horizontal = "right"

        # Calculate vertical position
        vertical = "in the middle"
        if cy < frame_height / 3:
            vertical = "high up"
        elif cy > 2 * frame_height / 3:
            vertical = "down low"

        # Calculate angle in degrees
        center_frame_x = frame_width / 2
        deg_x = abs(cx - center_frame_x)
        deg_y = cy
        
        if deg_x > 0:
            rad_theta = math.atan(deg_y / deg_x)
            deg_theta = rad_theta * (180 / math.pi)
            degrees = 90 - deg_theta
        else:
            degrees = 90

        # Calculate depth category
        depth_category = "unknown distance"
        if median_depth is not None:
            if median_depth < 0.35:
                depth_category = "far"
            else:
                depth_category = "within arms length"

        objects.append({
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "depth_norm": median_depth,
            "position": vertical + " " + horizontal,
            "depth_category": depth_category,
            "degrees": degrees,
            "horizontal": horizontal,
            "vertical": vertical
        })

    # Default values if no objects detected
    degrees = None
    horizontal = "center"
    vertical = "in the middle"
    depth_category = "unknown distance"
    bbox = [0,0,0,0]

    # Use first detected object for overall values
    if objects:
        degrees = objects[0]["degrees"]
        horizontal = objects[0]["horizontal"]
        vertical = objects[0]["vertical"]
        depth_category = objects[0]["depth_category"]
        bbox = objects[0]["bbox"]

    return objects, depth_map, frame, degrees, horizontal, vertical, depth_category, bbox


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