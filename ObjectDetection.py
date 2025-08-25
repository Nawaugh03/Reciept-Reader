import cv2
import pytesseract
from ultralytics import YOLO

# Load YOLO model (start from pretrained weights)
model = YOLO("yolov8n.pt")  # 'n' = nano, small and fast to train

# Train the model
model.train(
    data="Datasets/data.yaml",  # path to data.yaml
    epochs=50,                           # number of training epochs
    imgsz=640,                           # image size
    batch=16                             # adjust based on GPU
)

