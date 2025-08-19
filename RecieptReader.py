import cv2
import pytesseract
import pandas as pd

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Load the image
img_path = 'Recipts/Reciept1.jpg'
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found. Check the file path: {img_path}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding (binarize)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

# Save temp file for OCR
temp_path = "temp_receipt.jpg"
cv2.imwrite(temp_path, thresh)

