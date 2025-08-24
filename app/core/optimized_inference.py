try:
    import torch
    import onnx
    import onnxruntime as ort
    TORCH_AVAILABLE = True
    ONNX_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    ONNX_AVAILABLE = False

import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class InferenceBackend(ABC):
    """Abstract base class for different inference backends"""

    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """Load the model from the given path"""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image for inference"""
        pass

    @abstractmethod
    def infer(self, processed_image: np.ndarray) -> np.ndarray:
        """Run inference on the processed image"""
        pass

    @abstractmethod
    def postprocess(self, output: np.ndarray, original_shape: Tuple[int, int]) -> list:
        """Postprocess the model output"""
        pass

class PyTorchBackend(InferenceBackend):
    """PyTorch backend for YOLOv5 inference"""

    def __init__(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch.")
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_path: str) -> Any:
        """Load YOLOv5 model from PyTorch weights"""
        yolov5_path = str(Path(__file__).parent / 'models' / 'yolov5')
        self.model = torch.hub.load(yolov5_path, 'custom', path=model_path, source='local')
        self.model.to(self.device)
        self.model.eval()
        return self.model

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLOv5"""
        # YOLOv5 preprocessing is handled internally by the model
        return image

    def infer(self, processed_image: np.ndarray) -> np.ndarray:
        """Run inference"""
        with torch.no_grad():
            results = self.model(processed_image)
        return results.xyxy[0].cpu().numpy()

    def postprocess(self, output: np.ndarray, original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Postprocess YOLOv5 output"""
        detections = []
        for detection in output:
            x1, y1, x2, y2, conf, cls = detection
            class_name = str(int(cls))
            if self.model and hasattr(self.model, 'names'):
                class_name = self.model.names[int(cls)]
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class': int(cls),
                'class_name': class_name
            })
        return detections

class ONNXBackend(InferenceBackend):
    """ONNX Runtime backend for optimized inference"""

    def __init__(self):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available. Please install onnxruntime.")
        self.session = None
        self.input_name = None
        self.output_names = None

    def load_model(self, model_path: str) -> Any:
        """Load ONNX model"""
        # Convert .pt to .onnx if needed
        onnx_path = model_path.replace('.pt', '.onnx')
        if not Path(onnx_path).exists():
            logger.info(f"Converting {model_path} to ONNX format...")
            self._convert_to_onnx(model_path, onnx_path)

        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'GPU' in ort.get_device() else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        return self.session

    def _convert_to_onnx(self, pt_path: str, onnx_path: str):
        """Convert PyTorch model to ONNX format"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ONNX conversion")

        # Load the PyTorch model temporarily
        yolov5_path = str(Path(__file__).parent / 'models' / 'yolov5')
        model = torch.hub.load(yolov5_path, 'custom', path=pt_path, source='local')
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, 640, 640)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logger.info(f"Model converted to ONNX: {onnx_path}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX inference"""
        # Resize to 640x640
        resized = cv2.resize(image, (640, 640))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Transpose to CHW format
        chw = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(chw, axis=0)

        return batched

    def infer(self, processed_image: np.ndarray) -> np.ndarray:
        """Run ONNX inference"""
        if self.session is None:
            raise RuntimeError("Model not loaded")
        outputs = self.session.run(self.output_names, {self.input_name: processed_image})
        return outputs[0]  # Return first output

    def postprocess(self, output: np.ndarray, original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Postprocess ONNX output (simplified YOLOv5 format)"""
        detections = []

        # YOLOv5 ONNX output shape: [1, 25200, 85] for 640x640 input
        # 85 = 4 (bbox) + 1 (conf) + 80 (classes)
        output = output[0]  # Remove batch dimension

        for detection in output:
            # Extract bbox, confidence, and class scores
            bbox = detection[:4]
            conf = detection[4]
            class_scores = detection[5:]

            # Skip low confidence detections
            if conf < 0.25:
                continue

            # Get class with highest score
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id] * conf

            if class_conf < 0.25:
                continue

            # Convert bbox from center format to corner format
            x_center, y_center, width, height = bbox
            x1 = int((x_center - width/2) * original_shape[1] / 640)
            y1 = int((y_center - height/2) * original_shape[0] / 640)
            x2 = int((x_center + width/2) * original_shape[1] / 640)
            y2 = int((y_center + height/2) * original_shape[0] / 640)

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(class_conf),
                'class': int(class_id),
                'class_name': f'class_{class_id}'  # Will be mapped later
            })

        return detections

class OptimizedInferenceManager:
    """Manager class for different inference backends"""

    def __init__(self, backend_type: str = 'pytorch'):
        self.backend_type = backend_type
        self.backend = self._create_backend(backend_type)
        self.class_names = None

    def _create_backend(self, backend_type: str) -> InferenceBackend:
        """Create the appropriate backend"""
        if backend_type == 'pytorch':
            return PyTorchBackend()
        elif backend_type == 'onnx':
            return ONNXBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")

    def load_model(self, model_path: str, class_names: Optional[list] = None):
        """Load model with specified backend"""
        logger.info(f"Loading model with {self.backend_type} backend: {model_path}")
        self.backend.load_model(model_path)
        self.class_names = class_names

        # Set class names for backends that support it
        if isinstance(self.backend, PyTorchBackend) and self.backend.model and hasattr(self.backend.model, 'names'):
            if class_names:
                self.backend.model.names = class_names

    def predict(self, image: np.ndarray) -> list:
        """Run inference on an image"""
        original_shape = image.shape[:2]

        # Preprocess
        processed = self.backend.preprocess(image)

        # Infer
        output = self.backend.infer(processed)

        # Postprocess
        detections = self.backend.postprocess(output, original_shape)

        # Map class names if available
        if self.class_names:
            for det in detections:
                if det['class'] < len(self.class_names):
                    det['class_name'] = self.class_names[det['class']]

        return detections

    def get_inference_time(self, image: np.ndarray, num_runs: int = 10) -> float:
        """Benchmark inference time"""
        import time

        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.predict(image)
            end_time = time.time()
            times.append(end_time - start_time)

        return float(np.mean(times))

def create_inference_manager(backend_type: str = 'pytorch') -> OptimizedInferenceManager:
    """Factory function to create inference manager"""
    return OptimizedInferenceManager(backend_type)