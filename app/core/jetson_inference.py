#!/usr/bin/env python3
"""
Jetson Nano TensorRT Inference Engine
Optimized for NVIDIA Jetson Nano edge computing platform
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import json
import os

# Try to import TensorRT and CUDA dependencies
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError as e:
    TENSORRT_AVAILABLE = False
    # Create minimal dummy implementations
    trt = None
    cuda = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jetson_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TensorRTInference:
    """TensorRT inference engine optimized for Jetson Nano"""

    def __init__(self, engine_path: str, class_names: Optional[List[str]] = None):
        """
        Initialize TensorRT inference engine

        Args:
            engine_path: Path to the TensorRT engine file (.engine or .trt)
            class_names: List of class names for the model
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT is not available. Please install TensorRT for Jetson Nano.")

        self.engine_path = Path(engine_path)
        self.class_names = class_names or []
        self.engine = None
        self.context = None
        self.stream = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.input_shape = None
        self.output_shape = None

        # Initialize TensorRT
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize TensorRT engine and execution context"""
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT is not available. Please install TensorRT for Jetson Nano.")

        try:
            logger.info(f"Loading TensorRT engine from: {self.engine_path}")

            # Load engine file
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()

            # Create runtime
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

            # Deserialize engine
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)

            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")

            # Create execution context
            self.context = self.engine.create_execution_context()

            # Get input/output information
            self._setup_io_bindings()

            logger.info("TensorRT engine initialized successfully")
            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Output shape: {self.output_shape}")

        except Exception as e:
            logger.error(f"Failed to initialize TensorRT engine: {str(e)}")
            raise

    def _setup_io_bindings(self):
        """Setup input/output bindings for inference"""
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT is not available")

        # Get number of I/O tensors
        num_io_tensors = self.engine.num_io_tensors

        for i in range(num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)

            logger.info(f"Tensor {i}: {tensor_name}, shape: {tensor_shape}, dtype: {tensor_dtype}")

            # Allocate device memory
            size = trt.volume(tensor_shape)
            dtype = trt.nptype(tensor_dtype)

            # Create host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Store binding information
            binding = {
                'name': tensor_name,
                'shape': tensor_shape,
                'dtype': dtype,
                'host_mem': host_mem,
                'device_mem': device_mem,
                'size': size
            }

            self.bindings.append(device_mem)

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
                self.input_shape = tensor_shape
            else:
                self.outputs.append(binding)
                self.output_shape = tensor_shape

        # Create CUDA stream
        self.stream = cuda.Stream()

    def preprocess_image(self, image: np.ndarray, input_shape: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Preprocess image for TensorRT inference

        Args:
            image: Input image (BGR format)
            input_shape: Target input shape (width, height)

        Returns:
            Preprocessed image ready for inference
        """
        try:
            # Resize image
            resized = cv2.resize(image, input_shape)

            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1]
            normalized = rgb.astype(np.float32) / 255.0

            # Convert to CHW format
            chw = np.transpose(normalized, (2, 0, 1))

            # Add batch dimension
            batched = np.expand_dims(chw, axis=0)

            return batched

        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def postprocess_detections(self, output: np.ndarray, original_shape: Tuple[int, int],
                             conf_threshold: float = 0.25, nms_threshold: float = 0.45) -> List[Dict[str, Any]]:
        """
        Postprocess TensorRT output to extract detections

        Args:
            output: Raw model output
            original_shape: Original image shape (height, width)
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold for filtering overlapping boxes

        Returns:
            List of detections with bounding boxes, confidence scores, and class IDs
        """
        try:
            detections = []

            # YOLOv5 TensorRT output format: [batch, num_boxes, 85]
            # 85 = 4 (bbox) + 1 (conf) + 80 (classes)
            output = output[0]  # Remove batch dimension

            for detection in output:
                # Extract bbox, confidence, and class scores
                bbox = detection[:4]
                conf = detection[4]
                class_scores = detection[5:]

                # Skip low confidence detections
                if conf < conf_threshold:
                    continue

                # Get class with highest score
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id] * conf

                if class_conf < conf_threshold:
                    continue

                # Convert bbox from center format to corner format
                x_center, y_center, width, height = bbox

                # Scale to original image size
                x1 = int((x_center - width/2) * original_shape[1])
                y1 = int((y_center - height/2) * original_shape[0])
                x2 = int((x_center + width/2) * original_shape[1])
                y2 = int((y_center + height/2) * original_shape[0])

                # Ensure coordinates are within bounds
                x1 = max(0, min(x1, original_shape[1]))
                y1 = max(0, min(y1, original_shape[0]))
                x2 = max(0, min(x2, original_shape[1]))
                y2 = max(0, min(y2, original_shape[0]))

                # Get class name
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

                detection_dict = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(class_conf),
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                }

                detections.append(detection_dict)

            # Apply Non-Maximum Suppression
            detections = self._apply_nms(detections, nms_threshold)

            return detections

        except Exception as e:
            logger.error(f"Detection postprocessing failed: {str(e)}")
            return []

    def _apply_nms(self, detections: List[Dict[str, Any]], nms_threshold: float) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression to filter overlapping detections"""
        if len(detections) == 0:
            return detections

        # Sort by confidence score
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        filtered_detections = []

        for detection in detections:
            # Check if this detection overlaps with any already selected detection
            should_add = True

            for selected in filtered_detections:
                iou = self._calculate_iou(detection['bbox'], selected['bbox'])
                if iou > nms_threshold:
                    should_add = False
                    break

            if should_add:
                filtered_detections.append(detection)

        return filtered_detections

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on input image

        Args:
            image: Preprocessed input image

        Returns:
            Raw model output
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT is not available")

        try:
            # Copy input data to device
            np.copyto(self.inputs[0]['host_mem'], image.ravel())

            # Transfer input data to device
            cuda.memcpy_htod_async(
                self.inputs[0]['device_mem'],
                self.inputs[0]['host_mem'],
                self.stream
            )

            # Run inference
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )

            # Transfer output data to host
            cuda.memcpy_dtoh_async(
                self.outputs[0]['host_mem'],
                self.outputs[0]['device_mem'],
                self.stream
            )

            # Synchronize stream
            self.stream.synchronize()

            # Reshape output
            output = self.outputs[0]['host_mem'].reshape(self.output_shape)

            return output

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Complete prediction pipeline: preprocess -> infer -> postprocess

        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold for detections

        Returns:
            List of detections
        """
        try:
            # Store original shape
            original_shape = image.shape[:2]

            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Run inference
            output = self.infer(processed_image)

            # Postprocess detections
            detections = self.postprocess_detections(
                output, original_shape, conf_threshold
            )

            return detections

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return []

    def get_inference_time(self, image: np.ndarray, num_runs: int = 10) -> float:
        """Benchmark inference time"""
        times = []

        for _ in range(num_runs):
            start_time = time.time()
            _ = self.predict(image)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        logger.info(".3f")
        return float(avg_time)

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.stream:
                self.stream = None
            if self.context:
                del self.context
            if self.engine:
                del self.engine
            if self.runtime:
                del self.runtime

            logger.info("TensorRT resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

class JetsonVehicleDetector:
    """Complete vehicle detection system for Jetson Nano"""

    def __init__(self, vehicle_engine_path: str, plate_engine_path: Optional[str] = None):
        """
        Initialize Jetson vehicle detector

        Args:
            vehicle_engine_path: Path to vehicle detection TensorRT engine
            plate_engine_path: Path to license plate detection TensorRT engine (optional)
        """
        self.vehicle_detector = None
        self.plate_detector = None

        # Vehicle class names
        self.vehicle_classes = [
            "Ambulance", "Bus", "Car", "Motorcycle", "Truck"
        ]

        # Initialize vehicle detector
        try:
            self.vehicle_detector = TensorRTInference(
                vehicle_engine_path,
                self.vehicle_classes
            )
            logger.info("Vehicle detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize vehicle detector: {str(e)}")
            raise

        # Initialize license plate detector (optional)
        if plate_engine_path and Path(plate_engine_path).exists():
            try:
                self.plate_detector = TensorRTInference(plate_engine_path)
                logger.info("License plate detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize license plate detector: {str(e)}")

    def detect_vehicles(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Detect vehicles in the image"""
        if self.vehicle_detector is None:
            raise RuntimeError("Vehicle detector not initialized")

        return self.vehicle_detector.predict(image, conf_threshold)

    def detect_license_plates(self, vehicle_roi: np.ndarray, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Detect license plates in vehicle ROI"""
        if self.plate_detector is None:
            return []

        return self.plate_detector.predict(vehicle_roi, conf_threshold)

    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """Process a single frame for complete vehicle analysis"""
        try:
            # Detect vehicles
            vehicles = self.detect_vehicles(frame, conf_threshold)

            results = []

            for vehicle in vehicles:
                x1, y1, x2, y2 = vehicle['bbox']

                # Extract vehicle ROI
                vehicle_roi = frame[y1:y2, x1:x2]

                # Detect license plate (if available)
                license_plates = []
                if self.plate_detector:
                    license_plates = self.detect_license_plates(vehicle_roi, conf_threshold)

                # Create result entry
                result = {
                    'vehicle': vehicle,
                    'license_plates': license_plates,
                    'timestamp': time.time()
                }

                results.append(result)

            return {
                'detections': results,
                'frame_shape': frame.shape,
                'num_vehicles': len(vehicles)
            }

        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}")
            return {
                'detections': [],
                'frame_shape': frame.shape,
                'num_vehicles': 0,
                'error': str(e)
            }

    def cleanup(self):
        """Clean up all resources"""
        if self.vehicle_detector:
            self.vehicle_detector.cleanup()
        if self.plate_detector:
            self.plate_detector.cleanup()

def create_jetson_detector(vehicle_engine_path: str, plate_engine_path: Optional[str] = None) -> JetsonVehicleDetector:
    """Factory function to create Jetson vehicle detector"""
    return JetsonVehicleDetector(vehicle_engine_path, plate_engine_path)

# Example usage and testing functions
def test_jetson_inference():
    """Test function for Jetson inference"""
    try:
        # Initialize detector
        detector = create_jetson_detector(
            vehicle_engine_path="models/vehicle_detection.engine",
            plate_engine_path="models/license_plate_detection.engine"
        )

        # Load test image
        test_image = cv2.imread("test_image.jpg")
        if test_image is None:
            logger.error("Test image not found")
            return

        # Process frame
        results = detector.process_frame(test_image)

        logger.info(f"Detected {results['num_vehicles']} vehicles")

        # Benchmark performance
        if detector.vehicle_detector:
            inference_time = detector.vehicle_detector.get_inference_time(test_image, num_runs=20)
            logger.info(".1f")
        else:
            logger.warning("Vehicle detector not available for benchmarking")

        # Cleanup
        detector.cleanup()

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    # Run test if executed directly
    test_jetson_inference()