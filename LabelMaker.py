import cv2
import os
import re
import pytesseract
folder_name = "Receipts"
output_folder = "Labeled_pics"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.makedirs(output_folder, exist_ok=True)

# Globals
ref_point = []
cropping = False
coords = []
img_name = ""
newfilename=""
Online=True
max_width = 1200
max_height = 720
scale_x, scale_y = 1, 1  # Scaling factors
CustomerInc=1
datetimeInc=1
totalInc=1
tipInc=1
MiscInc=1
##############################################################################

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, coords

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # Draw rectangle on resized image
        cv2.rectangle(display_img, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", display_img)

        # Save scaled coordinates for original image
        x1, y1 = ref_point[0]
        x2, y2 = ref_point[1]
        orig_x1, orig_y1 = int(x1 * scale_x), int(y1 * scale_y)
        orig_x2, orig_y2 = int(x2 * scale_x), int(y2 * scale_y)

        coords.append(((orig_x1, orig_y1), (orig_x2, orig_y2)))

for filename in os.listdir(folder_name):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_name = filename
        path = os.path.join(folder_name, filename)
        original_img = cv2.imread(path)
        if original_img is None:
            print(f"Skipping {filename}: Not found")
            continue

        # Calculate resize scale
        h, w = original_img.shape[:2]
        aspect_ratio= w / h
        # Calculate new width and height while keeping aspect ratio
        if w / max_width > h / max_height:
            # Width is the limiting factor
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            # Height is the limiting factor
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        display_img = cv2.resize(original_img, (new_width, new_height))

        # Scaling factors
        scale_x = w / new_width
        scale_y = h / new_height

        coords.clear()
        cv2.namedWindow(f"{filename}")
        cv2.setMouseCallback(f"{filename}", click_and_crop)

        print(f"\n📸 Processing {filename}")
        print("Draw a rectangle. Press 'n' for next image, 'r' to reset, or 'q' to quit.")

        while Online:
            cv2.imshow(f"{filename}", display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):  # Reset
                display_img = cv2.resize(original_img, (new_width, new_height))
                coords.clear()
                print("Reset selections.")
            elif key == ord("n"):  # Next image
                break
            elif key == ord("q"):  # Quit
                Online = False
                break

        # Save cropped regions from original image Didnt saved the cropped images
        for i, ((x1, y1), (x2, y2)) in enumerate(coords):
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            crop = original_img[ymin:ymax, xmin:xmax]
            # OCR
            # Convert to grayscale for better OCR accuracy
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # Optional preprocessing for cleaner OCR
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            scale = 4  # Upscale by 3x
            resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            text = pytesseract.image_to_string(resized)
            print(f"OCR Result for crop {i+1}:\n{text.strip()}\n{'-'*50}")
            customer_pattern= r"(?:Order(?: by)?[:\s]+([A-Za-z]+))|(?:Card\s*#?.*?\n([A-Za-z][A-Za-z\s]+)(?=\nLoyalty))"
            datetime_pattern = r"(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM))"
            total_pattern = r"Total\s*\$?(\d+\.\s*\d{2})"
            tip_pattern = r"Tip[s]?\s*[:\-]?\s*\$?\s*([\d,]+\s*\.\s*\d{2})"

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
            #label = input(f"Enter label for crop {i+1} (Customer/Total/Date/Tip): ").strip()
            save_name = newfilename
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, crop)
            print(f"✅ Saved crop: {save_path}")

        if Online == False:
            cv2.destroyAllWindows()

"""
Sample idea for rectangle selection and OCR processing
# Store rectangles
rectangles = []



def line_select_callback(eclick, erelease):
    #""Callback when a rectangle is drawn""
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

"""

Very important!!! Sample 2 for reading and processing a single image
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