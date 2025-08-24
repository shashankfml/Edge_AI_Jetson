#!/usr/bin/env python3
"""
Benchmark script to compare different inference backends
Usage: python benchmark_inference.py --backend pytorch --backend onnx --image_path test_image.jpg
"""

import argparse
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_image(image_path: str, size: tuple = (640, 640)) -> np.ndarray:
    """Load and preprocess test image"""
    if not Path(image_path).exists():
        logger.error(f"Test image not found: {image_path}")
        raise FileNotFoundError(f"Test image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize to specified size
    image = cv2.resize(image, size)
    return image

def benchmark_backend(backend_name: str, model_path: str, test_image: np.ndarray,
                     num_runs: int = 10) -> Dict[str, Any]:
    """Benchmark a specific inference backend"""
    results = {
        'backend': backend_name,
        'success': False,
        'inference_times': [],
        'avg_inference_time': 0.0,
        'fps': 0.0,
        'error': None
    }

    try:
        # Import the optimized inference manager
        from app.core.optimized_inference import create_inference_manager

        # Create inference manager
        manager = create_inference_manager(backend_name)

        # Load model
        logger.info(f"Loading {backend_name} model: {model_path}")
        manager.load_model(model_path)

        # Warm-up run
        logger.info(f"Warm-up run for {backend_name}")
        _ = manager.predict(test_image)

        # Benchmark runs
        logger.info(f"Running {num_runs} benchmark runs for {backend_name}")
        inference_times = []
        detections = []

        for i in range(num_runs):
            start_time = time.time()
            current_detections = manager.predict(test_image)
            end_time = time.time()

            inference_time = end_time - start_time
            inference_times.append(inference_time)
            detections = current_detections  # Store last detection result
            logger.info(".3f")

        # Calculate statistics
        avg_time = np.mean(inference_times)
        fps = 1.0 / avg_time

        results.update({
            'success': True,
            'inference_times': inference_times,
            'avg_inference_time': avg_time,
            'fps': fps,
            'num_detections': len(detections) if detections else 0
        })

        logger.info(f"{backend_name} Results:")
        logger.info(".3f")
        logger.info(".2f")
        logger.info(f"  Detections: {results['num_detections']}")

    except Exception as e:
        logger.error(f"Error benchmarking {backend_name}: {str(e)}")
        results['error'] = str(e)

    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark inference backends')
    parser.add_argument('--backends', nargs='+', choices=['pytorch', 'onnx', 'tensorrt'],
                       default=['pytorch'], help='Backends to benchmark')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to test image')
    parser.add_argument('--model_path', type=str,
                       default='app/core/models/vehicle_detection/exp1/weights/best.pt',
                       help='Path to model file')
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of benchmark runs')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                       help='Output CSV file path')

    args = parser.parse_args()

    # Load test image
    test_image = load_test_image(args.image_path)
    if test_image is None:
        logger.error("Failed to load test image")
        return

    logger.info(f"Test image shape: {test_image.shape}")

    # Benchmark each backend
    all_results = []

    for backend in args.backends:
        logger.info(f"\n{'='*50}")
        logger.info(f"Benchmarking {backend.upper()} backend")
        logger.info(f"{'='*50}")

        results = benchmark_backend(backend, args.model_path, test_image, args.num_runs)
        all_results.append(results)

    # Save results to CSV
    successful_results = [r for r in all_results if r['success']]

    if successful_results:
        df = pd.DataFrame(successful_results)
        df.to_csv(args.output, index=False)
        logger.info(f"\nBenchmark results saved to: {args.output}")

        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"{'Backend':<12} {'Avg Time (ms)':<15} {'FPS':<8} {'Detections':<10}")
        print("-" * 60)

        for result in successful_results:
            avg_time_ms = result['avg_inference_time'] * 1000
            print(f"{result['backend']:<12} {avg_time_ms:<15.1f} {result['fps']:<8.1f} {result['num_detections']:<10}")

        # Find fastest backend
        fastest = min(successful_results, key=lambda x: x['avg_inference_time'])
        print(f"\nFastest backend: {fastest['backend']} ({fastest['fps']:.1f} FPS)")

    else:
        logger.error("No successful benchmarks to report")

if __name__ == "__main__":
    main()