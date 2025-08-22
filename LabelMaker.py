import cv2
import os

folder_name = "Receipts"
output_folder = "Labeled_pics"
os.makedirs(output_folder, exist_ok=True)

# Globals
ref_point = []
cropping = False
coords = []
img_name = ""
scale_x, scale_y = 0.75, 0.75  # Scaling factors

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
        new_width = 1000
        scale = new_width / w
        new_height = int(h * scale)
        display_img = cv2.resize(original_img, (new_width, new_height))

        # Scaling factors
        scale_x = w / new_width
        scale_y = h / new_height

        coords.clear()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop)

        print(f"\n📸 Processing {filename}")
        print("Draw a rectangle. Press 'n' for next image, 'r' to reset, or 'q' to quit.")

        while True:
            cv2.imshow("image", display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):  # Reset
                display_img = cv2.resize(original_img, (new_width, new_height))
                coords.clear()
                print("Reset selections.")
            elif key == ord("n"):  # Next image
                break
            elif key == ord("q"):  # Quit
                exit()

        # Save cropped regions from original image
        for i, ((x1, y1), (x2, y2)) in enumerate(coords):
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            crop = original_img[ymin:ymax, xmin:xmax]

            label = input(f"Enter label for crop {i+1} (Customer/Total/Date/Tip): ").strip()
            save_name = f"{os.path.splitext(filename)[0]}_{label}_{i+1}.jpg"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, crop)
            print(f"✅ Saved crop: {save_path}")

cv2.destroyAllWindows()
