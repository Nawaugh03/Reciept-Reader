import os
import cv2
import pytesseract
import pandas as pd


# If on Windows and PATH not set, manually add tesseract.exe path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Path to your folder
image_folder = "Labeled_pics"

# If on Windows and PATH is not set, set Tesseract executable path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Data storage
data = []

# Loop through all images in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        print(f"Processing: {filename}")
        
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale for better OCR accuracy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Optional preprocessing for cleaner OCR
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        scale = 3  # Upscale by 3x
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Extract text using pytesseract
        text = pytesseract.image_to_string(resized)

        # Extract only numbers (for totals, tips, etc.)
        custom_config = r'--oem 3 --psm 6 outputbase digits'
        numbers = pytesseract.image_to_string(resized, config=custom_config)

        # Append results
        data.append({
            "Filename": filename,
            "Extracted_Text": text.strip(),
            "Extracted_Numbers": numbers.strip()
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_csv = "receipt_extraction.csv"
df.to_csv(output_csv, index=False)

print(f"✅ Extraction completed! Data saved to {output_csv}")
