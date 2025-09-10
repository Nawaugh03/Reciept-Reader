import cv2
import csv
import os
import re
import pytesseract
from ultralytics import YOLO
import pandas as pd
from matplotlib import pyplot as plt
# Load YOLO model (start from pretrained weights)
print("Loading model...")
#model = YOLO("yolov8n.pt")  # 'n' = nano, small and fast to train

# Load your trained weights (best.pt)
model = YOLO("runs/detect/Fine_tuned_model8/weights/best.pt")

#model.eval()  # set model to evaluation mode

#Image to test
Image="Receipts/Receipt2.jpg"

# If on Windows and PATH not set, manually add tesseract.exe path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

custom_config_number = r'--oem 3 --psm 6 outputbase digits'
custom_config_text= r'--oem 3 --psm 6 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
"""
# Train the model
model.train(
    data="Datasets/data.yaml",  # path to data.yaml
    epochs=75,                           # number of training epochs
    imgsz=640,                           # image size
    batch=16,                          # batch size
    name="Fine_tuned_model",           # model name
    project="runs/detect"            # project name
)
"""
# Tuning the model for a bit
metrics = model.val()
print(metrics)
model.eval()
#print(model.names)

def get_results(Image_paths, df):
        # Prepare row dictionary with default None
         # Prepare row dictionary with defaults
        raw_data = {
        "Filename": Image_paths,
        "Customer_Name": None,
        "Total": None,
        "Tip": None,
        "DateTime": None
        }
        results = model(Image_paths, save=True, conf=0.5)
        # Get first result
        r = results[0]
        # Load cropped image
        img = cv2.imread(Image_paths)
        img_copy = img.copy()
        # Convert to grayscale
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

        # Resize (upscale) the cropped region
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Apply threshold (binarization)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Optional: remove noise with median blur
        denoised = cv2.medianBlur(thresh, 3)

        boxes = r.boxes  # Boxes object
        #print(boxes)
        for box in boxes:
            cls = int(box.cls[0])        # class ID
            conf = float(box.conf[0])    # confidence
            label = model.names[cls]  # class name (customer_name, total, tip, datetime)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            #print(f"Class {cls},label {label}, Conf {conf:.2f}, BBox {xyxy}")
            # Crop detected region
            x1, y1, x2, y2 = xyxy
            crop = img[y1:y2, x1:x2]
            # 🔹 Scale up the crop (e.g., 3x bigger)
            scale_factor = 3
            if crop.size > 0:  # prevent errors on empty crops
                crop = cv2.resize(crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

                # Optional preprocessing for OCR
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # OCR
                if label in ["Total", "Tip"]:
                    variable = pytesseract.image_to_string(gray, config=custom_config_number)
                    variable = re.sub(r'[^\d.]', '', variable)
                else:
                    variable = pytesseract.image_to_string(gray, config=custom_config_text)
            else:
                variable = "[empty crop]"

            print(f"{label}: {variable}, {conf:.2f}, {xyxy}")
            if label in raw_data:
                raw_data[label] = variable

            df.loc[len(df)] = raw_data
            
            # 🔹 Draw bounding box + label on the image
            #cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(img_copy, f"{label} {conf:.2f}", (x1, y1 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 🔹 Show with matplotlib
        #plt.figure(figsize=(12, 8))
        #plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        #plt.axis("off")
        #plt.show()

folder = "Receipts/"
images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".jpg", ".png", ".jpeg"))]
df = pd.DataFrame(columns=["Filename", "Customer_Name", "Total", "Tip", "DateTime"])
for Image_path in images:
    get_results(Image_path, df)
df.drop_duplicates(subset=["Filename"], keep="last", inplace=True)
# Extract numbers and sort
df["file_num"] = df["Filename"].str.extract(r'(\d+)').astype(int)
df = df.sort_values(by="file_num").drop(columns="file_num").reset_index(drop=True)

df.to_csv("Results.csv", index=False)
