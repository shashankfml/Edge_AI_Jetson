import os
import cv2
from pathlib import Path

# Set your paths
labels_dir = Path("vehicle_data_v2/valid/labels")
images_dir = Path("vehicle_data_v2/valid/images")
output_labels_dir = Path("vehicle_data_v2/valid/labels_yolo")
output_labels_dir.mkdir(parents=True, exist_ok=True)


def convert_polygon_to_yolo(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    image_path = images_dir / (txt_file.stem + ".jpg")  # or .png
    if not image_path.exists():
        print(f"Image not found for: {image_path}")
        return

    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]

    yolo_lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        coords = list(map(int, parts[:8]))
        class_name = parts[8]

        # Skip unwanted classes
        if class_name not in class_map:
            continue

        x_coords = coords[0::2]
        y_coords = coords[1::2]

        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        class_id = class_map[class_name]
        yolo_line = f"{class_id} {x_center} {y_center} {box_width} {box_height}\n"
        yolo_lines.append(yolo_line)

    # Write YOLO annotation if there's at least one object
    if yolo_lines:
        out_file = output_labels_dir / txt_file.name
        with open(out_file, 'w') as f:
            f.writelines(yolo_lines)

class_map = {
    'ambulance': 0,
    'bus': 1,
    'car': 2,
    'biker': 3,
    'truck': 4
}

for txt_file in labels_dir.glob("*.txt"):
    convert_polygon_to_yolo(txt_file)