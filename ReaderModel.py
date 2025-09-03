import cv2
import csv
import os
import re
import pytesseract
from ultralytics import YOLO
import pandas as pd
from matplotlib import pyplot as plt
# Load YOLO model (start from pretrained weights)
model = YOLO("yolov8n.pt")  # 'n' = nano, small and fast to train

# Load your trained weights (best.pt)
#model = YOLO("runs/detect/Fine_tuned_model/weights/best.pt")

#model.eval()  # set model to evaluation mode

#Image to test
Image="Receipts/Receipt12.jpg"

# If on Windows and PATH not set, manually add tesseract.exe path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

custom_config_number = r'--oem 3 --psm 6 outputbase digits'
custom_config_text= r'--oem 3 --psm 6 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Train the model
model.train(
    data="Datasets/data.yaml",  # path to data.yaml
    epochs=50,                           # number of training epochs
    imgsz=640,                           # image size
    batch=16,                             # adjust based on GPU memory
    project="runs/detect",  # save to project/name
    name="Fine_tuned_model"  # name of the training run
)

# Tuning the model for a bit
metrics = model.val()
print(metrics)
model.eval()
"""
results = model(Image, save=True, conf=0.5)


# Get first result
r = results[0]
# Load cropped image
img = cv2.imread(Image)
img_copy = img.copy()
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
        variable = pytesseract.image_to_string(gray, config="--psm 6")
    else:
        variable = "[empty crop]"
    print(f"{ label}: {variable} \n{conf:.2f}, {xyxy}")
    # 🔹 Draw bounding box + label on the image
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_copy, f"{label} {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 🔹 Show with matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()



# Run on one image
results = model(Image, save=True, conf=0.5)  


#Extract the predictions
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])                # class id
        label = model.names[cls_id]             # class name (customer_name, total, tip, datetime)
        conf = float(box.conf[0])               # confidence
        xyxy = box.xyxy[0].tolist()             # [x1, y1, x2, y2]
        
        print(f"{label}: {conf:.2f}, {xyxy}")


#get results

# Run on one image
#results = model(Image, save=True, conf=0.5)  

variable=0
#fields = ['Customer_Name', 'DateTime', 'Tip', 'Total']
#df=pd.DataFrame(columns=fields)
folder_name= "Receipts"
# Get all image paths with valid extensions
#image_paths = [os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
image_paths = [Image]
for img_path in image_paths:
    img = cv2.imread(img_path)
    results = model(img,save=True,conf=0.5)
    print(f"Processing image: {img_path}\n")
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            # Crop detected region
            x1, y1, x2, y2 = xyxy
            crop = img[y1:y2, x1:x2]
            if label == "Tip" or label == "Total":
                variable=pytesseract.image_to_string(crop,config=custom_config_number)
            elif label == "Customer_Name":
                cleaned_text=[]
                variable=pytesseract.image_to_string(crop,config=custom_config_text)
                for line in variable.split('\n'):
                    if re.search(r"(Card\s*#|Loyalty\s*Member|Order\s*:)", line, re.IGNORECASE):
                        continue  # Skip lines with Card # or Loyalty Member
                    if line.strip():  # Keep non-empty lines
                        cleaned_text.append(line.strip())
                variable = ' '.join(cleaned_text)
            else:
                variable=pytesseract.image_to_string(crop, config=custom_config_text).strip()
            conf = float(box.conf[0])
            print(f"{ label}: {variable} \n{conf:.2f}, {xyxy}")
            


# Save Reuslts to CSV
fields = ['Customer_Name', 'DateTime', 'Tip', 'Total']
data = {f: "" for f in fields}

for r in results:
    for box in r.boxes:
        label = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        crop = img[y1:y2, x1:x2]
        text = pytesseract.image_to_string(crop).strip()
        data[label] = text

# Write to CSV
with open("receipts.csv", "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    if f.tell() == 0:  # write header only once
        writer.writeheader()
    writer.writerow(data)
"""