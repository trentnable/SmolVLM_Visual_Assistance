import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO
import urllib.request
import sys, os

import matplotlib.pyplot as plt

# ----------- YOLO SETUP -----------
yolo_model = YOLO("yolov8n.pt")  # "n" is nano model, good for Jetson

# ----------- MiDaS SETUP -----------
# Load MiDaS (small model is fast & accurate enough for real-time)

model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

def get_depth_map(frame):
    """Run MiDaS to get depth map for the frame."""
    input_batch = transform(frame).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    plt.imshow(output)
    plt.show()
    # Normalize depth to meters-ish (relative)
    depth = (output - output.min()) / (output.max() - output.min() + 1e-6)
    return depth

def fuse_yolo_midas(frame):
    """Run YOLO + MiDaS and return structured object list."""
    depth_map = get_depth_map(frame)
    results = yolo_model(frame, verbose=False)
    objects = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = results[0].names[int(box.cls)]
        confidence = float(box.conf)

        # Guard against empty crop
        if x2 > x1 and y2 > y1:
            depth_crop = depth_map[y1:y2, x1:x2]
            median_depth = float(np.median(depth_crop))
        else:
            median_depth = None

        objects.append({
            "label": label,
            "confidence": round(confidence, 3),
            "bbox": [x1, y1, x2, y2],
            "depth_norm": round(median_depth, 3) if median_depth is not None else None
        })

    return objects

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection + depth fusion
        objects = fuse_yolo_midas(frame)

        # Serialize to JSON
        json_output = json.dumps(objects, indent=2)
        print(json_output)

        # OPTIONAL: draw bounding boxes for visualization
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{obj['label']} {obj['depth_norm']:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

        cv2.imshow("YOLO + MiDaS", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
