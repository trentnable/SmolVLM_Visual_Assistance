from fastapi import FastAPI
from objectify import classify_request
from vision import fuse_yolo_midas, to_base64_img, setup_yolo, setup_midas
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI()


# Mode 1
out1 = classify_request("Where is my water bottle?")
yolo_model, class_names = setup_yolo("yolo11n.pt")
midas, transform = setup_midas("MiDaS_small")

@app.post("/detect")
async def detect(request: Request):
    # decode request
    # run fuse_yolo_midas()
    # return JSON
