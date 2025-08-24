# 🚗 Vehicle Detection System with MLOps Pipeline

A comprehensive vehicle detection system built with YOLOv5, featuring real-time video processing, license plate recognition, and a complete MLOps pipeline with model versioning, monitoring, and continuous learning capabilities.

## 🎯 Project Overview

This project implements an end-to-end vehicle detection system that processes video streams to identify vehicles, extract license plates, and determine vehicle colors. The system is designed with MLOps best practices including automated training pipelines, model versioning with DVC, drift monitoring, and containerized deployment.

### Key Features

- **Real-time Vehicle Detection**: YOLOv5-based detection for multiple vehicle types (Ambulance, Bus, Car, Motorcycle, Truck)
- **License Plate Recognition**: OCR-powered text extraction using EasyOCR
- **Vehicle Color Detection**: K-means clustering for color identification
- **Duplicate Prevention**: Smart tracking system to avoid duplicate detections
- **Web Interface**: Streamlit-based dashboard for video upload and processing
- **MLOps Pipeline**: Complete pipeline with DVC, MLflow, and automated retraining
- **Performance Optimization**: Multi-backend support (PyTorch, ONNX, TensorRT)
- **Containerized Deployment**: Docker support for consistent deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  Core Detection │────│   ML Models     │
│                 │    │                 │    │                 │
│ • Video Upload  │    │ • Vehicle       │    │ • YOLOv5        │
│ • Settings      │    │   Processor     │    │ • License Plate │
│ • Results View  │    │ • OCR Engine    │    │ • Color         │
│ • Feedback      │    │ • Tracking      │    │   Detection     │
└─────────────────┘    │ • Optimization  │    │                 │
                       └─────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌─────────────────┐
                       │   MLOps Stack   │
                       │                 │
                       │ • DVC Pipeline  │
                       │ • MLflow        │
                       │ • Drift Monitor │
                       │ • Auto Retrain  │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)
- Docker (optional)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd app_project_jetson
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run final_app.py
```

4. **Access the web interface**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
app_project_jetson/
├── app/
│   └── core/
│       ├── detection.py          # Main vehicle detection logic
│       ├── jetson_inference.py   # Jetson Nano optimized inference
│       ├── optimized_inference.py # Multi-backend inference manager
│       └── models/               # Trained model artifacts
├── config/
│   └── inference_config.yaml     # Inference configuration
├── scripts/
│   ├── benchmark_inference.py    # Performance benchmarking
│   ├── jetson_inference_example.py # Jetson examples
│   ├── retrain.sh               # Automated retraining script
│   └── update_classes.py        # Class management utilities
├── monitor/
│   └── monitor_drift.py         # Model drift detection
├── feedback/
│   ├── images/                  # User feedback images
│   └── labels/                  # Corrected labels
├── Docker_creation/             # Docker deployment files
├── final_app.py                 # Main Streamlit application
├── train_with_mlflow.py         # MLflow training pipeline
├── dvc.yaml                     # DVC pipeline configuration
├── requirements.txt             # Python dependencies
└── Dockerfile                   # Container configuration
```

## 🎮 Usage

### Web Interface

1. **Upload Video**: Select a video file (MP4, MOV, AVI)
2. **Configure Settings**: 
   - Adjust confidence threshold (0.1-1.0)
   - Set frame processing interval (1-10)
3. **Process Video**: Click "Process Video" to start analysis
4. **View Results**: 
   - Real-time processing display
   - Detection statistics
   - Downloadable CSV/Excel reports
5. **Provide Feedback**: Correct misclassifications for model improvement

### Command Line

```bash
# Run training pipeline
python train_with_mlflow.py --params params_v1.yaml

# Monitor model drift
python monitor/monitor_drift.py

# Benchmark inference performance
python scripts/benchmark_inference.py --backends pytorch onnx

# Run automated retraining
bash scripts/retrain.sh
```

### Python API

```python
from app.core.detection import VehicleProcessor

# Initialize processor
processor = VehicleProcessor()

# Process video
results = processor.process_video(
    input_path="video.mp4",
    confidence_thresh=0.5,
    process_every=3
)

# Access results
print(f"Detected {len(results['dataframe'])} vehicles")
```

## ⚙️ Configuration

### Inference Settings

Edit `config/inference_config.yaml`:

```yaml
inference:
  backend: 'pytorch'  # 'pytorch', 'onnx', 'tensorrt'
  confidence_threshold: 0.5
  vehicle_model_path: 'app/core/models/vehicle_detection/exp1/weights/best.pt'
  plate_model_path: 'app/core/models/license_data_pyimagesearch/yolov5s_size640_epochs5_batch32_small/weights/best.pt'
  enable_fp16: true
  device: 'auto'
```

### Training Parameters

Configure training in `params_v1.yaml`, `params_v2.yaml`, etc.:

```yaml
epochs: 50
batch_size: 16
img_size: 640
learning_rate: 0.01
```

## 🔧 MLOps Pipeline

### Data Version Control (DVC)

```bash
# Track data versions
dvc add data/vehicle_data_v1
dvc add data/vehicle_data_v2

# Run training pipeline
dvc repro train_v1

# Monitor drift
dvc repro monitor
```

### MLflow Integration

```bash
# Start MLflow UI
mlflow ui

# View experiments at http://localhost:5000
```

### Automated Retraining

The system automatically triggers retraining when:
- Model drift is detected
- Sufficient feedback data is collected
- Performance metrics drop below threshold

## 🚢 Deployment

### Docker Deployment

```bash
# Build image
docker build -t vehicle-detection .

# Run container
docker run -p 8501:8501 vehicle-detection

# Run with GPU support
docker run --gpus all -p 8501:8501 vehicle-detection
```

### Jetson Nano Deployment

```bash
# Use Jetson-optimized inference
python scripts/jetson_inference_example.py

# Configure for edge deployment
# Edit config/inference_config.yaml for Jetson-specific settings
```

## 📊 Performance Optimization

### Backend Comparison

| Backend | Speed | Memory | Compatibility |
|---------|-------|--------|---------------|
| PyTorch | Medium | High | Excellent |
| ONNX | Fast | Medium | Good |
| TensorRT | Fastest | Low | NVIDIA Only |

### Benchmarking

```bash
# Compare inference backends
python scripts/benchmark_inference.py \
    --backends pytorch onnx \
    --image_path test_image.jpg \
    --num_runs 100
```

## 🔍 Monitoring & Maintenance

### Model Drift Detection

The system continuously monitors:
- Prediction confidence distributions
- Class distribution changes
- Performance metrics over time

### Logging

All processing activities are logged to `vehicle_process.log`:
- Detection results
- Performance metrics
- Error tracking
- User feedback

## 🧪 Testing

```bash
# Run comprehensive tests
python final_test.py

# Test specific components
python test.py
```

## 📈 Results Export

The system generates detailed reports including:
- Vehicle detection counts by type
- License plate information
- Color analysis
- Processing timestamps
- Confidence scores

Export formats:
- CSV files
- Excel spreadsheets
- JSON data

## 🔄 Continuous Learning

The feedback system enables continuous model improvement:

1. **User Corrections**: Users can correct misclassifications
2. **Data Collection**: Feedback is automatically stored
3. **Automated Retraining**: System retrains when sufficient data is available
4. **Model Updates**: New models are deployed automatically

## 🛠️ Troubleshooting

### Common Issues

**Memory Issues**
```bash
# Reduce batch size in config
batch_size: 1

# Use CPU inference
device: 'cpu'
```

**Performance Issues**
```bash
# Enable GPU acceleration
device: 'cuda'

# Use optimized backend
backend: 'onnx'
```

**Model Loading Errors**
```bash
# Check model paths in config/inference_config.yaml
# Ensure models exist in specified directories
```

## 📋 Requirements

### System Requirements
- Python 3.9+
- 4GB+ RAM
- CUDA-compatible GPU (recommended)
- 10GB+ storage space

### Key Dependencies
- PyTorch 1.13.1
- Ultralytics YOLOv5
- OpenCV
- EasyOCR
- Streamlit
- MLflow
- DVC
- ONNX Runtime

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for object detection
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition
- [Streamlit](https://streamlit.io/) for the web interface
- [MLflow](https://mlflow.org/) for experiment tracking
- [DVC](https://dvc.org/) for data version control

---

**Built with ❤️ for intelligent vehicle monitoring systems**
