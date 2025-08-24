#!/usr/bin/env python3
"""
Jetson Nano Inference Example
Demonstrates how to use the TensorRT inference engine for vehicle detection
"""

import cv2
import numpy as np
import logging
import time
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(level)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function demonstrating Jetson inference usage"""
    parser = argparse.ArgumentParser(description='Jetson Nano Vehicle Detection Example')
    parser.add_argument('--engine_path', type=str, required=True,
                       help='Path to TensorRT engine file (.engine)')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_path', type=str, default='output.jpg',
                       help='Path to save output image')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='Confidence threshold for detections')

    args = parser.parse_args()

    try:
        # Import the Jetson inference module
        from app.core.jetson_inference import create_jetson_detector

        logger.info("Initializing Jetson vehicle detector...")
        logger.info(f"Engine path: {args.engine_path}")
        logger.info(f"Image path: {args.image_path}")

        # Create detector (this will raise an error if TensorRT is not available)
        detector = create_jetson_detector(args.engine_path)

        # Load and process image
        if not Path(args.image_path).exists():
            raise FileNotFoundError(f"Image not found: {args.image_path}")

        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {args.image_path}")

        logger.info(f"Loaded image with shape: {image.shape}")

        # Run inference
        logger.info("Running inference...")
        start_time = time.time()

        results = detector.process_frame(image, conf_threshold=args.conf_threshold)

        end_time = time.time()
        inference_time = end_time - start_time

        logger.info(f"Inference time: {inference_time:.3f} seconds")
        logger.info(f"Detected {results['num_vehicles']} vehicles")

        # Draw detections on image
        output_image = image.copy()
        for detection in results['detections']:
            vehicle = detection['vehicle']
            x1, y1, x2, y2 = vehicle['bbox']

            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{vehicle['class_name']} {vehicle['confidence']:.2f}"
            cv2.putText(output_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw license plate if detected
            if detection['license_plates']:
                for plate in detection['license_plates']:
                    px1, py1, px2, py2 = plate['bbox']
                    cv2.rectangle(output_image, (px1, py1), (px2, py2), (255, 0, 0), 2)

        # Save output image
        cv2.imwrite(args.output_path, output_image)
        logger.info(f"Output saved to: {args.output_path}")

        # Print detailed results
        print("\n" + "="*50)
        print("DETECTION RESULTS")
        print("="*50)
        print(f"Image: {args.image_path}")
        print(f"Inference time: {inference_time:.3f} seconds")
        print(f"Vehicles detected: {results['num_vehicles']}")

        for i, detection in enumerate(results['detections']):
            vehicle = detection['vehicle']
            print(f"\nVehicle {i+1}:")
            print(f"  Class: {vehicle['class_name']}")
            print(f"  Confidence: {vehicle['confidence']:.3f}")
            print(f"  Bounding box: {vehicle['bbox']}")
            if detection['license_plates']:
                print(f"  License plates: {len(detection['license_plates'])} detected")

        # Cleanup
        detector.cleanup()
        logger.info("Cleanup completed")

    except ImportError as e:
        logger.error(f"TensorRT not available: {e}")
        logger.error("Please install TensorRT for Jetson Nano:")
        logger.error("  sudo apt-get update")
        logger.error("  sudo apt-get install tensorrt")
        logger.error("  pip install pycuda")
        return 1

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0

def benchmark_jetson_inference():
    """Benchmark function for performance testing"""
    try:
        from app.core.jetson_inference import create_jetson_detector

        # Initialize detector
        detector = create_jetson_detector("models/vehicle_detection.engine")

        # Load test image
        test_image = cv2.imread("test_image.jpg")
        if test_image is None:
            logger.error("Test image not found")
            return

        # Benchmark
        num_runs = 20
        logger.info(f"Running {num_runs} benchmark runs...")

        times = []
        for i in range(num_runs):
            start_time = time.time()
            results = detector.process_frame(test_image)
            end_time = time.time()
            times.append(end_time - start_time)
            logger.info(".3f")

        avg_time = np.mean(times)
        fps = 1.0 / avg_time

        logger.info("Benchmark Results:")
        logger.info(f"Average inference time: {avg_time:.3f} seconds")
        logger.info(f"FPS: {fps:.1f}")

        detector.cleanup()

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")

if __name__ == "__main__":
    exit(main())