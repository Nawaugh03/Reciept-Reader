import cv2
import re
import os
import pytesseract
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import pandas as pd

# If on Windows and PATH not set, manually add tesseract.exe path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the image
folder_name= "Receipts"
imaage_path=[os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
imaages=[]

# Create output folder for crops
output_folder = "Labeled_pics"
os.makedirs(output_folder, exist_ok=True)

"""
# Read image
img_path = 'Receipts/Reciept1.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for img_path in imaage_path:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found. Check the file path: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imaages.append(img_rgb)

"""


