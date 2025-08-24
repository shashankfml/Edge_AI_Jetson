import sklearn
from sklearn.cluster import KMeans
import cv2
import easyocr
import torch
import numpy as np
import csv
from datetime import datetime, timedelta
import os
import io
import time
from collections import OrderedDict, defaultdict
import pandas as pd
from pathlib import Path
import logging

# Initialize logging
logging.basicConfig(
    filename='vehicle_process.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CORE_DIR = Path(__file__).parent.resolve()

class VehicleTracker:
    def __init__(self, max_age=timedelta(seconds=5)):
        self.vehicles = OrderedDict()
        self.max_age = max_age

    def update(self, vehicle_id, position, plate_text):
        current_time = datetime.now()
        self.vehicles[vehicle_id] = {
            'position': position,
            'plate': plate_text,
            'last_seen': current_time
        }
        logger.debug(f"Updated tracker for {vehicle_id} with plate {plate_text} at {position}")

        for vid in list(self.vehicles.keys()):
            if (current_time - self.vehicles[vid]['last_seen']) > self.max_age:
                logger.debug(f"Removing stale vehicle {vid}")
                self.vehicles.pop(vid)

    def is_duplicate(self, position, plate_text, distance_thresh=50):
        for vid, data in self.vehicles.items():
            if plate_text != "NOT_FOUND" and data['plate'] == plate_text:
                logger.debug(f"Duplicate found by plate: {plate_text}")
                return True
            prev_pos = data['position']
            distance = np.linalg.norm(np.array(position) - np.array(prev_pos))
            if distance < distance_thresh:
                logger.debug(f"Duplicate found by position: {position}")
                return True
        return False

class VehicleProcessor:
    def __init__(self):
        logger.info("Initializing VehicleProcessor")
        self.load_models()
        self.reader = easyocr.Reader(['en'])
        self.tracker = VehicleTracker()
        self.COLORS = {
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
        self.drift_detector = DriftDetector()
        self.known_classes = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]

    def load_models(self):
        yolov5_path = str(CORE_DIR / 'models' / 'yolov5')
        vehicle_weights = str(CORE_DIR / 'models' / 'vehicle_detection' / 'exp1' / 'weights' / 'best.pt')
        plate_weights = str(CORE_DIR / 'models' / 'license_data_pyimagesearch' / 'yolov5s_size640_epochs5_batch32_small' / 'weights' / 'best.pt')
        
        self.vehicle_model = torch.hub.load(yolov5_path, 'custom', path=vehicle_weights, source='local')
        self.plate_model = torch.hub.load(yolov5_path, 'custom', path=plate_weights, source='local')
        self.vehicle_model.classes = [0, 1, 2, 3, 4]
        logger.info("Models loaded successfully")

    def get_vehicle_color(self, roi):
        h, w = roi.shape[:2]
        roi = roi[h//4:3*h//4, w//4:3*w//4]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
        mask = hsv[:,:,2] > 50
        if not np.any(mask):
            return [0, 0, 0]
        pixels = roi[mask].reshape(-1, 3)
        kmeans = KMeans(n_clusters=1).fit(pixels)
        return kmeans.cluster_centers_[0].astype(int)

    def closest_color(self, rgb):
        min_dist = float('inf')
        color_name = "unknown"
        for name, color_rgb in self.COLORS.items():
            dist = np.linalg.norm(rgb - color_rgb)
            if dist < min_dist:
                min_dist = dist
                color_name = name
        return color_name

    def preprocess_plate(self, plate_roi):
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def process_video(self, input_path, display_frame=None, progress_bar=None, confidence_thresh=0.5, process_every=3):
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results, unknown_detections, all_detections = [], [], []
        frame_count = 0

        logger.info(f"Processing video: {input_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % process_every != 0:
                continue

            vehicle_results = self.vehicle_model(frame)

            viz_frame = frame.copy()
            for detection in vehicle_results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = map(float, detection)
                if conf < confidence_thresh:
                    continue

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                vehicle_roi = frame[y1:y2, x1:x2]
                vehicle_type = self.vehicle_model.names[int(cls)]

                if vehicle_type not in self.known_classes:
                    logger.warning(f"Unknown vehicle type detected: {vehicle_type}")
                    unknown_detections.append({
                        'image': vehicle_roi.copy(),
                        'predicted_class': vehicle_type,
                        'position': ((x1+x2)//2, (y1+y2)//2),
                        'timestamp': datetime.now(),
                        'confidence': conf
                    })
                    continue
                else:
                    # logger.info(f"All vehicle type detected")
                    # logger.info(f'Vehicle type: {vehicle_type}')
                    all_detections.append({
                        'image': vehicle_roi.copy(),
                        'predicted_class': vehicle_type,
                        'position': ((x1+x2)//2, (y1+y2)//2),
                        'timestamp': datetime.now(),
                        'confidence': conf
                    })


                dominant_color = self.get_vehicle_color(vehicle_roi)
                vehicle_color = self.closest_color(dominant_color)

                plate_results = self.plate_model(vehicle_roi)
                license_number = "UNREADABLE"
          
                for plate_det in plate_results.xyxy[0]:
                    px1, py1, px2, py2, p_conf, _ = map(float, plate_det)
                    if p_conf < 0.4:
                        continue
                    px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])
                    plate_roi = vehicle_roi[py1:py2, px1:px2]

                    try:
                        processed_plate = self.preprocess_plate(plate_roi)
                        ocr_results = self.reader.readtext(processed_plate, detail=1)
                        if ocr_results:
                            best_result = max(ocr_results, key=lambda x: x[2])
                            if best_result[2] > 0.7:
                                license_number = best_result[1].upper().replace(" ", "")
                    except Exception as e:
                        logger.error(f"OCR failed: {e}")
                        license_number = "OCR_ERROR"

                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(viz_frame,
                            f"{vehicle_type} | {vehicle_color} | {license_number}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)  
                                            
                results.append([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    vehicle_type,
                    vehicle_color,
                    license_number
                ])
                vehicle_center = ((x1+x2)//2, (y1+y2)//2)
                if self.tracker.is_duplicate(vehicle_center, license_number):
                    continue

                self.tracker.update(f"{x1}_{y1}_{x2}_{y2}", vehicle_center, license_number)


                # cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(viz_frame,
                #             f"{vehicle_type} | {vehicle_color} | {license_number}",
                #             (x1, y1-10),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.7, (0, 255, 0), 2)                

            if display_frame:
                #viz_frame = frame.copy()
                display_frame.image(viz_frame, channels="BGR")
                #cv2.imshow('Vehicle Tracking',viz_frame)

            if progress_bar:
                progress_bar.progress(frame_count / total_frames)
            
            time.sleep(0.03)

        cap.release()
        misclassifications = self._check_misclassifications(all_detections)

        if unknown_detections or misclassifications:
            logger.warning("Found Drift!")
            self._handle_drift(misclassifications)

        return {
            'dataframe': pd.DataFrame(results, columns=["Timestamp", "Vehicle_Type", "Color", "License_Plate"]),
            'csv_data': self._df_to_csv(results),
            'unknown_detections': unknown_detections,
            'misclassifications': misclassifications
        }

    def _df_to_csv(self, results):
        csv_buffer = io.StringIO()
        pd.DataFrame(results).to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()

    def _handle_drift(self, misclassifications):
        logger.info("Drift handling...")
        drift_results = self.drift_detector.check_drift(misclassifications)
        if drift_results['new_classes']:
            logger.warning(f"New vehicle classes detected: {drift_results['new_classes']}")
        if drift_results['misclassifications']:
            logger.warning(f"Misclassifications detected: {drift_results['misclassifications']}")

    def _check_misclassifications(self, detections):
        misclassifications = []
        logger.info("Checking for misclassification...")
        for det in detections:
            if det['predicted_class'] == 'Motorcycle':
                aspect_ratio = det['image'].shape[1] / det['image'].shape[0]
                is_bicycle = (
                    aspect_ratio < 0.8 or
                    self._detect_pedals(det['image'])
                )
                if is_bicycle:
                    logger.debug("Misclassified data is captured.")
                    misclassifications.append({
                        'actual_class': 'Bicycle',
                        'predicted_class': 'Motorcycle',
                        'image': det['image'],
                        'confidence': det['confidence'],
                        'reason': 'aspect_ratio' if aspect_ratio < 0.8 else 'pedals_detected'
                    })
        return misclassifications

    def _detect_pedals(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (0.4 < x/image.shape[1] < 0.6 and 
                y/image.shape[0] > 0.7 and
                0.05 < w/image.shape[1] < 0.15 and
                0.8 < h/w < 1.2):
                return True
        return False

class DriftDetector:
    def __init__(self):
        self.known_classes = {"Ambulance", "Bus", "Car", "Motorcycle", "Truck"}
        self.common_misclassifications = {
            ('Bicycle', 'Motorcycle'): {
                'threshold': 0.3,
                'features': ['aspect_ratio', 'pedal_presence']
            }
        }
        self.min_samples = 5

    def check_drift(self, detections):
        results = {
            'new_classes': defaultdict(int),
            'misclassifications': defaultdict(int)
        }
        for det in detections:
            if det['predicted_class'] not in self.known_classes:
                results['new_classes'][det['predicted_class']] += 1
            for (actual, predicted), config in self.common_misclassifications.items():
                if self._matches_misclassification(det, actual, predicted):
                    results['misclassifications'][(actual, predicted)] += 1

        return {
            'new_classes': [cls for cls, cnt in results['new_classes'].items() if cnt >= self.min_samples],
            'misclassifications': {
                pair: cnt for pair, cnt in results['misclassifications'].items()
                if cnt >= self.min_samples and cnt / sum(1 for d in detections if d['predicted_class'] in pair) > self.common_misclassifications[pair]['threshold']
            }
        }

    def _matches_misclassification(self, detection, actual, predicted):
        if detection['predicted_class'] != predicted:
            return False
        if (actual, predicted) == ('Bicycle', 'Motorcycle'):
            aspect_ratio = detection['image'].shape[1] / detection['image'].shape[0]
            return aspect_ratio < 0.8 or VehicleProcessor()._detect_pedals(detection['image'])
        return False



# import cv2
# import easyocr
# import torch
# import numpy as np
# from sklearn.cluster import KMeans
# import csv
# from datetime import datetime, timedelta
# import os
# import io
# from collections import OrderedDict, defaultdict
# import pandas as pd
# from pathlib import Path
# import logging

# # Initialize logging
# logging.basicConfig(
#     filename='vehicle_process.log',
#     filemode='w',
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# CORE_DIR = Path(__file__).parent.resolve()

# class VehicleTracker:
#     def __init__(self, max_age=timedelta(seconds=5)):
#         self.vehicles = OrderedDict()
#         self.max_age = max_age

#     def update(self, vehicle_id, position, plate_text):
#         current_time = datetime.now()
#         self.vehicles[vehicle_id] = {
#             'position': position,
#             'plate': plate_text,
#             'last_seen': current_time
#         }
#         logger.debug(f"Updated tracker for {vehicle_id} with plate {plate_text} at {position}")

#         for vid in list(self.vehicles.keys()):
#             if (current_time - self.vehicles[vid]['last_seen']) > self.max_age:
#                 logger.debug(f"Removing stale vehicle {vid}")
#                 self.vehicles.pop(vid)

#     def is_duplicate(self, position, plate_text, distance_thresh=50):
#         for vid, data in self.vehicles.items():
#             if plate_text != "NOT_FOUND" and data['plate'] == plate_text:
#                 logger.debug(f"Duplicate found by plate: {plate_text}")
#                 return True
#             prev_pos = data['position']
#             distance = np.linalg.norm(np.array(position) - np.array(prev_pos))
#             if distance < distance_thresh:
#                 logger.debug(f"Duplicate found by position: {position}")
#                 return True
#         return False

# class VehicleProcessor:
#     def __init__(self):
#         logger.info("Initializing VehicleProcessor")
#         self.load_models()
#         self.reader = easyocr.Reader(['en'])
#         self.tracker = VehicleTracker()
#         self.COLORS = {
#             'white': [255, 255, 255],
#             'black': [0, 0, 0],
#             'gray': [128, 128, 128],
#             'silver': [192, 192, 192],
#             'red': [255, 0, 0],
#             'blue': [0, 0, 255],
#             'green': [0, 128, 0],
#             'yellow': [255, 255, 0],
#             'brown': [165, 42, 42],
#         }
#         self.drift_detector = DriftDetector()
#         self.known_classes = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]

#     def load_models(self):
#         yolov5_path = str(CORE_DIR / 'models' / 'yolov5')
#         vehicle_weights = str(CORE_DIR / 'models' / 'vehicle_detection' / 'exp1' / 'weights' / 'best.pt')
#         plate_weights = str(CORE_DIR / 'models' / 'license_data_pyimagesearch' / 'yolov5s_size640_epochs5_batch32_small' / 'weights' / 'best.pt')
        
#         self.vehicle_model = torch.hub.load(yolov5_path, 'custom', path=vehicle_weights, source='local')
#         self.plate_model = torch.hub.load(yolov5_path, 'custom', path=plate_weights, source='local')
#         self.vehicle_model.classes = [0, 1, 2, 3, 4]
#         logger.info("Models loaded successfully")

#     def get_vehicle_color(self, roi):
#         h, w = roi.shape[:2]
#         roi = roi[h//4:3*h//4, w//4:3*w//4]
#         hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#         hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
#         mask = hsv[:,:,2] > 50
#         if not np.any(mask):
#             return [0, 0, 0]
#         pixels = roi[mask].reshape(-1, 3)
#         kmeans = KMeans(n_clusters=1).fit(pixels)
#         return kmeans.cluster_centers_[0].astype(int)

#     def closest_color(self, rgb):
#         min_dist = float('inf')
#         color_name = "unknown"
#         for name, color_rgb in self.COLORS.items():
#             dist = np.linalg.norm(rgb - color_rgb)
#             if dist < min_dist:
#                 min_dist = dist
#                 color_name = name
#         return color_name

#     def preprocess_plate(self, plate_roi):
#         gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#         _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         return thresh

#     def process_video(self, input_path, display_frame=None, progress_bar=None, confidence_thresh=0.5, process_every=3):
#         cap = cv2.VideoCapture(input_path)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         results, unknown_detections, all_detections = [], [], []
#         frame_count = 0

#         logger.info(f"Processing video: {input_path}")

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_count += 1
#             if frame_count % process_every != 0:
#                 continue

#             vehicle_results = self.vehicle_model(frame)

#             for detection in vehicle_results.xyxy[0]:
#                 x1, y1, x2, y2, conf, cls = map(float, detection)
#                 if conf < confidence_thresh:
#                     continue

#                 x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#                 vehicle_roi = frame[y1:y2, x1:x2]
#                 vehicle_type = self.vehicle_model.names[int(cls)]

#                 if vehicle_type not in self.known_classes:
#                     logger.warning(f"Unknown vehicle type detected: {vehicle_type}")
#                     unknown_detections.append({
#                         'image': vehicle_roi.copy(),
#                         'predicted_class': vehicle_type,
#                         'position': ((x1+x2)//2, (y1+y2)//2),
#                         'timestamp': datetime.now(),
#                         'confidence': conf
#                     })
#                     continue
#                 else:
#                     # logger.info(f"All vehicle type detected")
#                     # logger.info(f'Vehicle type: {vehicle_type}')
#                     all_detections.append({
#                         'image': vehicle_roi.copy(),
#                         'predicted_class': vehicle_type,
#                         'position': ((x1+x2)//2, (y1+y2)//2),
#                         'timestamp': datetime.now(),
#                         'confidence': conf
#                     })


#                 dominant_color = self.get_vehicle_color(vehicle_roi)
#                 vehicle_color = self.closest_color(dominant_color)

#                 plate_results = self.plate_model(vehicle_roi)
#                 license_number = "UNREADABLE"


#                 for plate_det in plate_results.xyxy[0]:
#                     px1, py1, px2, py2, p_conf, _ = map(float, plate_det)
#                     if p_conf < 0.4:
#                         continue
#                     px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])
#                     plate_roi = vehicle_roi[py1:py2, px1:px2]

#                     try:
#                         processed_plate = self.preprocess_plate(plate_roi)
#                         ocr_results = self.reader.readtext(processed_plate, detail=1)
#                         if ocr_results:
#                             best_result = max(ocr_results, key=lambda x: x[2])
#                             if best_result[2] > 0.7:
#                                 license_number = best_result[1].upper().replace(" ", "")
#                     except Exception as e:
#                         logger.error(f"OCR failed: {e}")
#                         license_number = "OCR_ERROR"

#                 results.append([
#                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     vehicle_type,
#                     vehicle_color,
#                     license_number
#                 ])
#                 vehicle_center = ((x1+x2)//2, (y1+y2)//2)
#                 if self.tracker.is_duplicate(vehicle_center, license_number):
#                     continue

#                 self.tracker.update(f"{x1}_{y1}_{x2}_{y2}", vehicle_center, license_number)

#                 if display_frame:
#                     viz_frame = frame.copy()
#                     cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(viz_frame,
#                                f"{vehicle_type} | {vehicle_color} | {license_number}",
#                                (x1, y1-10),
#                                cv2.FONT_HERSHEY_SIMPLEX,
#                                0.7, (0, 255, 0), 2)
#                     display_frame.image(viz_frame, channels="BGR")

#             if progress_bar:
#                 progress_bar.progress(frame_count / total_frames)

#         cap.release()
#         misclassifications = self._check_misclassifications(all_detections)

#         if unknown_detections or misclassifications:
#             logger.warning("Found Drift!")
#             self._handle_drift(misclassifications)

#         return {
#             'dataframe': pd.DataFrame(results, columns=["Timestamp", "Vehicle_Type", "Color", "License_Plate"]),
#             'csv_data': self._df_to_csv(results),
#             'unknown_detections': unknown_detections,
#             'misclassifications': misclassifications
#         }

#     def _df_to_csv(self, results):
#         csv_buffer = io.StringIO()
#         pd.DataFrame(results).to_csv(csv_buffer, index=False)
#         return csv_buffer.getvalue()

#     def _handle_drift(self, misclassifications):
#         logger.info("Drift handling...")
#         drift_results = self.drift_detector.check_drift(misclassifications)
#         if drift_results['new_classes']:
#             logger.warning(f"New vehicle classes detected: {drift_results['new_classes']}")
#         if drift_results['misclassifications']:
#             logger.warning(f"Misclassifications detected: {drift_results['misclassifications']}")

#     def _check_misclassifications(self, detections):
#         misclassifications = []
#         logger.info("Checking for misclassification...")
#         for det in detections:
#             if det['predicted_class'] == 'Motorcycle':
#                 aspect_ratio = det['image'].shape[1] / det['image'].shape[0]
#                 is_bicycle = (
#                     aspect_ratio < 0.8 or
#                     self._detect_pedals(det['image'])
#                 )
#                 if is_bicycle:
#                     logger.debug("Misclassified data is captured.")
#                     misclassifications.append({
#                         'actual_class': 'Bicycle',
#                         'predicted_class': 'Motorcycle',
#                         'image': det['image'],
#                         'confidence': det['confidence'],
#                         'reason': 'aspect_ratio' if aspect_ratio < 0.8 else 'pedals_detected'
#                     })
#         return misclassifications

#     def _detect_pedals(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 50, 150)
#         contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#         for cnt in contours:
#             x, y, w, h = cv2.boundingRect(cnt)
#             if (0.4 < x/image.shape[1] < 0.6 and 
#                 y/image.shape[0] > 0.7 and
#                 0.05 < w/image.shape[1] < 0.15 and
#                 0.8 < h/w < 1.2):
#                 return True
#         return False

# class DriftDetector:
#     def __init__(self):
#         self.known_classes = {"Ambulance", "Bus", "Car", "Motorcycle", "Truck"}
#         self.common_misclassifications = {
#             ('Bicycle', 'Motorcycle'): {
#                 'threshold': 0.3,
#                 'features': ['aspect_ratio', 'pedal_presence']
#             }
#         }
#         self.min_samples = 5

#     def check_drift(self, detections):
#         results = {
#             'new_classes': defaultdict(int),
#             'misclassifications': defaultdict(int)
#         }
#         for det in detections:
#             if det['predicted_class'] not in self.known_classes:
#                 results['new_classes'][det['predicted_class']] += 1
#             for (actual, predicted), config in self.common_misclassifications.items():
#                 if self._matches_misclassification(det, actual, predicted):
#                     results['misclassifications'][(actual, predicted)] += 1

#         return {
#             'new_classes': [cls for cls, cnt in results['new_classes'].items() if cnt >= self.min_samples],
#             'misclassifications': {
#                 pair: cnt for pair, cnt in results['misclassifications'].items()
#                 if cnt >= self.min_samples and cnt / sum(1 for d in detections if d['predicted_class'] in pair) > self.common_misclassifications[pair]['threshold']
#             }
#         }

#     def _matches_misclassification(self, detection, actual, predicted):
#         if detection['predicted_class'] != predicted:
#             return False
#         if (actual, predicted) == ('Bicycle', 'Motorcycle'):
#             aspect_ratio = detection['image'].shape[1] / detection['image'].shape[0]
#             return aspect_ratio < 0.8 or VehicleProcessor()._detect_pedals(detection['image'])
#         return False

