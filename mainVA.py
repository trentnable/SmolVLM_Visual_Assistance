from fastapi import FastAPI, Request
from objectify import classify_request
from vision import fuse_yolo_midas, to_base64_img, setup_yolo, setup_midas
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


start_time = time.time()

# Mode 1
out1 = int(classify_request("Where is my water bottle?"))
yolo_model, class_names = setup_yolo("yolo11n.pt")

print("! ! ! ! !\n" + yolo_model.names[(out1)] + "\n! ! ! ! !")

midas, transform = setup_midas("MiDaS_small")

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")


@app.post("/detect")
async def detect(request: Request):
    data = await request.json()
    image_b64 = data.get("image")
    if not image_b64:
        return JSONResponse(content={"error": "No image provided"}, status_code=400)

    # Decode base64 → OpenCV frame
    image_bytes = base64.b64decode(image_b64.split(",")[1])  # strip data header
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    objects, depth_map, frame_with_boxes = fuse_yolo_midas(frame, yolo_model, midas, transform, class_id=out1)

    return {
        "objects": objects,
        "frame_b64": to_base64_img(frame_with_boxes),
        "depth_b64": to_base64_img(depth_map)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
