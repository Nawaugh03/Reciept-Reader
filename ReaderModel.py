import cv2
import csv
import shutil, os
import re
import pytesseract
from ultralytics import YOLO
import pandas as pd
from matplotlib import pyplot as plt
import glob

class RRModel(): #Receipt Reader Model class that stores the model and its functionality to be used in a web app
    def __init__(self):
        self.default_model = "yolov8n.pt"
        self.model_type= self.get_latest_best_pt()
        self.model = self.load_model()
        self.custom_config_number = r'--oem 3 --psm 6 outputbase digits'
        self.custom_config_text= r'--oem 3 --psm 6 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self.pytesseract=pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        self.df = pd.DataFrame(columns=["Filename", "Customer_Name", "Total", "Tip", "Datetime"])
        self.folder = "Receipts/"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.images = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.endswith((".jpg", ".png", ".jpeg"))]
    def get_latest_best_pt(self, base_dir="runs/detect/"):
        best_pts = glob.glob(os.path.join(base_dir, "**", "weights", "best.pt"), recursive=True)
        if not best_pts:
            return self.default_model  # Return default if no trained models found
        latest = max(best_pts, key=os.path.getmtime)
        latest = latest.replace("/", "\\")  # For Windows compatibility
        return latest

    def load_model(self):
        print("Loading model...")
        model = YOLO(self.model_type)
        return model
    
    def train_model(self,data_path="Datasets/data.yaml", epochs=75, imgsz=640, batch=16, name="Fine_tuned_model", project="runs/detect"):
        # Train the model
        self.model = YOLO(self.default_model)  # Reload default model for training
        self.model.train(
            data=data_path,  # path to data.yaml
            epochs=epochs,                           # number of training epochs
            imgsz=imgsz,                           # image size
            batch=batch,                          # batch size
            name=name,           # model name
            project=project            # project name
            )
        self.modeltype = self.get_latest_best_pt()  # Update to the newly trained model
        self.model = self.load_model()  # Reload the model with new weights
    
    def format_usd(self, value):
        value = re.sub(r'[^\d]', '', value)  # Remove non-digit characters
        if not value:
            return ""
        # If value is all digits, treat last two as cents
        if len(value) > 2:
            amount = float(value[:-2] + '.' + value[-2:])
        else:
            amount = float('0.' + value.zfill(2))
        return f"${amount:,.2f}"
        
    def evaluate_model(self):
        metrics = self.model.val()
        print(metrics)
        self.model.eval()
        return metrics
    def get_results(self, Image_paths):
        # Prepare row dictionary with default None
         # Prepare row dictionary with defaults
        raw_data = {
        "Filename": Image_paths,
        "Customer_Name": None,
        "Total": "$0.00",
        "Tip": "$0.00",
        "Datetime": None
        }
        results = self.model(Image_paths, 
                        save=False,
                        project="output",
                        name="latest",
                        conf=0.5)
        # Get first result
        r = results[0]
        img = cv2.imread(Image_paths)
        boxes = r.boxes  # Boxes object
        #print(boxes)
        for box in boxes:
            cls = int(box.cls[0])        # class ID
            conf = float(box.conf[0])    # confidence
            label = self.model.names[cls].replace("_", " ").title().replace(" ", "_")# class name (customer_name, total, tip, datetime)
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
                    variable = pytesseract.image_to_string(gray, config=self.custom_config_number)
                    variable = self.format_usd(variable)
                else:
                    variable = pytesseract.image_to_string(gray, config=self.custom_config_text)
                    if label == "Customer_Name":
                        # Remove all non-letter characters (keeps spaces if you want)
                        variable = re.sub(r'[^A-Za-z ]+', '', variable)
                        # Optionally, strip leading/trailing whitespace and collapse multiple spaces
                        variable = ' '.join(variable.split())
                    if label == "Datetime":
                        # Keep only digits, slashes, dashes, colons, and spaces
                        variable = re.sub(r'[^0-9/\-: ]+', '', variable)
                        variable = variable.replace('\n', '').replace('\r', '')
                        # Optionally, strip leading/trailing whitespace and collapse multiple spaces
                        variable = ' '.join(variable.split())
            else:
                variable = "[empty crop]"

            print(f"{label}: {variable}, {conf:.2f}, {xyxy}")
            if label in raw_data:
                raw_data[label] = variable

        self.df.loc[len(self.df)] = raw_data

    def process_all_images(self):
        for Image_path in self.images:
            self.get_results(Image_path)
        self.df.drop_duplicates(subset=["Filename"], keep="last", inplace=True)
        # Extract numbers and sort
        self.df["file_num"] = self.df["Filename"].str.extract(r'(\d+)').astype(int)
        self.df = self.df.sort_values(by="file_num").drop(columns="file_num").reset_index(drop=True)

    def save_results_csv(self, filename="Results.csv"):
        self.df.to_csv(filename, index=False)
        # Now delete YOLO's output folder
        pred_path = "runs/detect/predict"
        if os.path.exists(pred_path):
            shutil.rmtree(pred_path)
    def save_results_excel(self, filename="Results.xlsx"):
        self.df.to_excel(filename, index=False)
        # Now delete YOLO's output folder
        pred_path = "runs/detect/predict"
        if os.path.exists(pred_path):
            shutil.rmtree(pred_path)

if __name__ == "__main__":
    rrmodel = RRModel()
    print(rrmodel.model_type)
    #model = YOLO("runs/detect/Fine_tuned_model8/weights/best.pt")
    rrmodel.process_all_images()
    print(rrmodel.df)
    #rrmodel.save_results()
    


"""
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
#Training model
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
        results = model(Image_paths, 
                        save=False,
                        project="output",
                        name="latest",
                        conf=0.5)
        # Get first result
        r = results[0]
        img = cv2.imread(Image_paths)
        boxes = r.boxes  # Boxes object
        #print(boxes)
        for box in boxes:
            cls = int(box.cls[0])        # class ID
            conf = float(box.conf[0])    # confidence
            label = model.names[cls].strip().title().replace("_", "") # class name (customer_name, total, tip, datetime)
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

# Now delete YOLO's output folder
pred_path = "runs/detect/predict"
if os.path.exists(pred_path):
    shutil.rmtree(pred_path)
"""
#--- END IGNORE ---