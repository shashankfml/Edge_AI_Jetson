import cv2
import easyocr
import torch
import numpy as np
from sklearn.cluster import KMeans
import csv
from datetime import datetime
from pathlib import Path

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load YOLOv5 models with error handling
try:
    vehicle_model = torch.hub.load('./yolov5', 'custom', source='local', 
                                 path='/home/fmlpc/app_project/vehicles_data_pyimagesearch/yolov5s_size640_epochs5_batch32_small/weights/best.pt')
    plate_model = torch.hub.load('./yolov5', 'custom', source='local',
                               path='/home/fmlpc/app_project/license_data_pyimagesearch/yolov5s_size640_epochs5_batch32_small/weights/best.pt')
    
    # Set vehicle classes to detect (car=2, bus=5, truck=7)
    vehicle_model.classes = [2, 5, 7]
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Color definitions
COLORS = {
    'white': [255, 255, 255],
    'black': [0, 0, 0],
    'gray': [128, 128, 128],
    'silver': [192, 192, 192],
    'red': [255, 0, 0],
    'blue': [0, 0, 255],
    'green': [0, 128, 0],
    'yellow': [255, 255, 0],
    'brown': [165, 42, 42],
}

def get_vehicle_color(roi):
    """Get dominant color from vehicle ROI"""
    # Crop center to avoid edges
    h, w = roi.shape[:2]
    roi = roi[h//4:3*h//4, w//4:3*w//4]
    
    # Convert to HSV and equalize
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    
    # Mask out dark pixels
    v = hsv[:,:,2]
    mask = v > 50
    if not np.any(mask):
        return [0, 0, 0]
    
    # Get dominant color from non-dark areas
    pixels = roi[mask].reshape(-1, 3)
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_[0].astype(int)

def closest_color(rgb):
    """Find closest named color"""
    min_dist = float('inf')
    color_name = "unknown"
    for name, color_rgb in COLORS.items():
        dist = np.linalg.norm(rgb - color_rgb)
        if dist < min_dist:
            min_dist = dist
            color_name = name
    return color_name

def preprocess_plate(plate_roi):
    """Enhance license plate for OCR"""
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Initialize CSV file
csv_file = "vehicle_data.csv"
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Vehicle_Type", "Vehicle_Color", "License_Plate"])

# Process video
video_path = "/home/fmlpc/Shashank/Course_Work/AA_project_MLOPS/Stock_test_videos/8321860-hd_1920_1080_30fps.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
process_every_n_frames = 3  # Process every 3rd frame for better performance

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        continue
    
    # Vehicle detection
    vehicle_results = vehicle_model(frame)
    
    for detection in vehicle_results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = map(float, detection)
        if conf < 0.5:  # Confidence threshold
            continue
            
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        vehicle_roi = frame[y1:y2, x1:x2]
        vehicle_type = vehicle_model.names[int(cls)]
        
        # Get vehicle color
        dominant_color = get_vehicle_color(vehicle_roi)
        vehicle_color = closest_color(dominant_color)
        
        # License plate detection
        plate_results = plate_model(vehicle_roi)
        license_number = "NOT_FOUND"
        
        for plate_det in plate_results.xyxy[0]:
            px1, py1, px2, py2, p_conf, _ = map(float, plate_det)
            if p_conf < 0.4:  # Plate confidence threshold
                continue
                
            px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])
            plate_roi = vehicle_roi[py1:py2, px1:px2]
            
            # OCR processing
            try:
                processed_plate = preprocess_plate(plate_roi)
                ocr_result = reader.readtext(processed_plate, detail=0)
                if ocr_result:
                    license_number = ocr_result[0].upper().replace(" ", "")
            except Exception as e:
                license_number = f"OCR_ERROR: {str(e)}"
                continue
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, vehicle_type, vehicle_color, license_number])
            
            # Visualization (optional)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, 
            #            f"{vehicle_type} | {vehicle_color} | {license_number}",
            #            (x1, y1-10), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 
            #            0.7, (0, 255, 0), 2)
    
    # Display output (optional)
    # cv2.imshow("Vehicle Detection", frame)
    # if cv2.waitKey(1) == ord('q'):
    #     break

cap.release()
# cv2.destroyAllWindows()