import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO
import urllib.request
import sys, os

import matplotlib.pyplot as plt

# API stuff for connecting to html
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
from io import BytesIO
from PIL import Image

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# text to speech
from gtts import gTTS
import subprocess

# input classification script
import classify

# voice to text
# import asyncio
# import whisper

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # define the device

# define classification model
# classification_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# tokenizer = AutoTokenizer.from_pretrained(classification_model)
# model = AutoModelForCausalLM.from_pretrained(classification_model).to(device)

# define yolo model
yolo_model = YOLO("yolo11n.pt")  # "n" is nano model, good for Jetson

# define midas model
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type) # load midas

midas.to(device) # send to device
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

    depth = (output - output.min()) / (output.max() - output.min() + 1e-6)
    return depth

def fuse_yolo_midas(frame, best_index, object_name, user_request):
    """Run classification → YOLO → MiDaS, and return structured result."""

    # 2. Run depth estimation
    depth_map = get_depth_map(frame)

    # 3. Run YOLO for only the predicted class
    results = yolo_model(frame, classes=[best_index], verbose=False)

    objects = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = results[0].names[int(box.cls)]
        confidence = float(box.conf)

        # depth crop for the bounding box
        depth_crop = depth_map[y1:y2, x1:x2]
        median_depth = float(np.median(depth_crop)) if depth_crop.size > 0 else None

        # bounding box overlay
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        depth_label = f"{label} ({median_depth:.2f})" if median_depth else label
        cv2.putText(frame, depth_label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # position classification
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        h, w = frame.shape[:2]
        horizontal = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
        vertical = "top" if cy < h / 3 else "bottom" if cy > 2 * h / 3 else "middle"

        distance = None
        if median_depth is not None:
            distance = "within arm's reach" if median_depth > 0.6 else "out of reach"

        objects.append({
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "depth_norm": median_depth,
            "position": f"{vertical} {horizontal}",
            "depth_category": distance,
            "classification_confidence": confidence,
            "object_name": object_name,
            "user_request": user_request
        })

    return objects, depth_map, frame

def to_base64_img(img):
    if len(img.shape) == 2:  # grayscale (depth)
        img = (img * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    # encode regardless (grayscale already converted to color above)
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")
    
@app.post("/detect")
async def detect(request: Request):
    data = await request.json()
    image_b64 = data.get("image")
    user_request = data.get("text", "")

    if not image_b64:
        return JSONResponse(content={"error": "No image provided"}, status_code=400)

    # Decode base64 to OpenCV frame
    image_bytes = base64.b64decode(image_b64.split(",")[1])  # strip header
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Interpret user's natural language query
    best_index, best_label, confidence = classify.classify_request(user_request)
    object_name = classify.label_texts[best_label].split(" ")[5]  # e.g. "banana"
    print(f"Interpreted request as class {best_label} ({object_name}), confidence={confidence:.3f}")

    
    objects, depth_map, frame_with_boxes = fuse_yolo_midas(frame, best_index, object_name, user_request)
    # print(objects)

    if not objects:
        description = f"Couldn't detect a {object_name}"

    else:
        parts = []
        for obj in objects:
            pos = obj.get("position", "somewhere")
            depth_category = obj.get("depth_category","")
            parts.append(f"{obj['label']} at the {pos} ({depth_category})")
        description = f"{len(objects)} {objects[0]['object_name']}s detected: " + ", ".join(parts) + "."

    return {
        "objects": objects,
        "frame_b64": to_base64_img(frame_with_boxes),
        "depth_b64": to_base64_img(depth_map),
        "interpreted_request": user_request,
        "spoken_response": description
    }

@app.post("/speak")
async def speak(request: Request):
    data = await request.json()
    string_to_speak = data.get("text", "")

    buf = BytesIO()
    gTTS(string_to_speak, lang="en", tld="co.uk").write_to_fp(buf)
    buf.seek(0)

    p = subprocess.Popen(
        [r"C:\Users\lmgre\Documents\SIU\Senior Design\pranjal_repo\SmolVLM_Visual_Assistance\FFmpeg\ffplay.exe", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
        stdin=subprocess.PIPE,
    )

    p.stdin.write(buf.read())
    p.stdin.close()
    p.wait()

    return 0

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)  # localhost only
