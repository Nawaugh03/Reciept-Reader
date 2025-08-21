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
img_path = 'Recipts/Reciept1.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create output folder for crops
output_folder = "crops"
os.makedirs(output_folder, exist_ok=True)

# Store rectangles
rectangles = []

def line_select_callback(eclick, erelease):
    """Callback when a rectangle is drawn"""
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    rectangles.append((x1, y1, x2, y2))
    print(f"Rectangle selected: ({x1}, {y1}) -> ({x2}, {y2})")

def toggle_selector(event):
    if event.key in ['Q', 'q']:  # Press Q to quit
        plt.close()

# Create interactive plot
fig, ax = plt.subplots()
ax.imshow(img_rgb)
toggle = RectangleSelector(
    ax, line_select_callback,
    useblit=True, button=[1], minspanx=5, minspany=5,
    spancoords='pixels', interactive=True
)

plt.connect('key_press_event', toggle_selector)
plt.show()

# Process selected regions
for i, (x1, y1, x2, y2) in enumerate(rectangles, start=1):
    crop = img[y1:y2, x1:x2]

    # Save crop as image
    crop_filename = os.path.join(output_folder, f"crop_{i}.png")
    cv2.imwrite(crop_filename, crop)

    # OCR
    text = pytesseract.image_to_string(crop)

    print(f"Region {i} saved as {crop_filename}")
    print(f"OCR Result:\n{text.strip()}\n{'-'*50}")
"""
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found. Check the file path: {img_path}")


resized = cv2.resize(img, (2000, 1500))

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 3)  # reduce noise
thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)


# OCR with Tesseract
custom_config = r'--oem 3 --psm 6'
receipt_text = pytesseract.image_to_string(thresh, config=custom_config)
if not receipt_text.strip():
    raise ValueError("No text detected in the image. Check the image quality or OCR settings.")
else:
    print("RAW OCR TEXT:\n", receipt_text)
    customer = re.search(r"User:\s*(\w+)", receipt_text)
    order   = re.search(r"Order:\s*(\w+)", receipt_text)
    time    = re.search(r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)", receipt_text)
    total   = re.search(r"Total\s+([\d\.]+)", receipt_text)
    tip     = re.search(r"Tip\s+([\d\.]+)", receipt_text, re.IGNORECASE)
    print("\nEXTRACTED FIELDS:")
    print("Customer:", customer.group(1) if customer else None)
    print("Time:", time.group(1) if time else None)
    print("Total:", total.group(1) if total else None)
    print("Tip:", tip.group(1) if tip else None)
"""