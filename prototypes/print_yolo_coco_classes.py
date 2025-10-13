from ultralytics import YOLO
from pprint import pprint

yolo_model = YOLO("yolo11n.pt")

# Pretty print the class index-to-name mapping
pprint(yolo_model.names)
