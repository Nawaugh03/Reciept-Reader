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
"""
for img_path in imaage_path:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found. Check the file path: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imaages.append(img_rgb)


# Store rectangles
rectangles = []
"""
# Function to handle rectangle selection
This functions is not working properly, it is not properly selecting the area.
"""
def line_select_callback(eclick, erelease):
    """Callback when a rectangle is drawn"""
    global current_img, coords
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    # Ensure proper ordering
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    rectangles.append((xmin, ymin, xmax, ymax))
     # Draw rectangle on Matplotlib image
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                         fill=False, color='red', linewidth=2)
    ax.add_patch(rect)
    plt.draw()

    print(f"Selected region: ({xmin}, {ymin}) -> ({xmax}, {ymax})")


def toggle_selector(event):
    if event.key in ['Q', 'q']:  # Press Q to quit
        plt.close()
for img_rgb in imaages[0:1]:
    print(img_rgb.name)
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
    CustomerInc=1
    datetimeInc=1
    totalInc=1
    tipInc=1
    MiscInc=1
    ##############################################################################
    # Iterate through rectangles and perform OCR
    for i, (x1, y1, x2, y2) in enumerate(rectangles):
        crop = img[y1:y2, x1:x2]
        label=""
        newfilename=""
        # Save crop as image
        crop_filename = os.path.join(output_folder, f"crop_image.png")
        cv2.imwrite(crop_filename, crop)

        # OCR
        text = pytesseract.image_to_string(crop)

        customer_pattern= r"(?:Order(?: by)?[:\s]+([A-Za-z]+))|(?:Card.*?\n([A-Za-z]+\s+[A-Za-z]+))"
        datetime_pattern = r"(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM))"
        total_pattern = r"Total\s+\$?([\d]+\.\d{2})"
        tip_pattern = r"Tip\s+\$?([\d]+\.\d{2})"

        customer_match = re.search(customer_pattern, text, re.IGNORECASE)
        datetime_match = re.search(datetime_pattern, text)
        total_match = re.search(total_pattern, text, re.IGNORECASE)
        tip_match = re.search(tip_pattern, text, re.IGNORECASE)

        customer = customer_match.group(1) if customer_match and customer_match.group(1) else (customer_match.group(2) if customer_match else None)
        datetime = datetime_match.group(1) if datetime_match else None
        total = total_match.group(1) if total_match else None
        tip = tip_match.group(1) if tip_match else None

        if customer_match:
            label="Customer"
            newfilename = f"{label}_{CustomerInc}.png"
            CustomerInc += 1
        elif datetime_match:
            label="DateTime"
            newfilename = f"{label}_{datetimeInc}.png"
            datetimeInc += 1
        elif total_match:
            label="Total"
            newfilename = f"{label}_{totalInc}.png"
            totalInc += 1
        elif tip_match:
            label="Tip"
            newfilename = f"{label}_{tipInc}.png"
            tipInc += 1
        else:
            label="Misc"
            newfilename = f"{label}_{MiscInc}.png"
            MiscInc += 1
            
        save_path = os.path.join(output_folder, newfilename)
        os.rename(crop_filename, save_path)
        
        print(f"Region {i} saved as {crop_filename}")
        #print(f"OCR Result:\n{text.strip()}\n{'-'*50}")

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