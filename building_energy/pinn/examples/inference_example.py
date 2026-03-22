"""
PINN Inference Example

This script demonstrates how to use a trained ThermalPINN model for
temperature prediction.

Usage:
    python inference_example.py --model models/pinn/thermal_model.onnx \
                                --outdoor-temp 30 \
                                --solar-radiation 800
"""

import os
import sys
import argparse
import logging
import json

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from building_energy.pinn import ThermalPINN, RoomParams, ExternalConditions
from building_energy.pinn.export import NPUAdapter, ONNXExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run PINN inference')
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/pinn/thermal_model.onnx',
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model config JSON'
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        default='onnxruntime',
        choices=['onnxruntime', 'ascend', 'mlu'],
        help='Inference backend'
    )
    
    parser.add_argument(
        '--outdoor-temp',
        type=float,
        default=30.0,
        help='Outdoor temperature (°C)'
    )
    
    parser.add_argument(
        '--solar-radiation',
        type=float,
        default=800.0,
        help='Solar radiation (W/m²)'
    )
    
    parser.add_argument(
        '--wind-speed',
        type=float,
        default=5.0,
        help='Wind speed (m/s)'
    )
    
    parser.add_argument(
        '--time',
        type=float,
        default=14.0,
        help='Time of day (0-24h)'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmark'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization'
    )
    
    return parser.parse_args()


def load_model_config(config_path: str) -> dict:
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_inference(args):
    """Run inference with the trained model."""
    logger.info("=" * 60)
    logger.info("ThermalPINN Inference Example")
    logger.info("=" * 60)
    
    # Load configuration
    if args.config is None:
        args.config = args.model.replace('.onnx', '_config.json')
    
    if not os.path.exists(args.config):
        logger.warning(f"Config file not found: {args.config}")
        logger.info("Using default room parameters")
        room_params = RoomParams(
            wall_thickness=0.2,
            room_volume=50.0,
            ac_position=(2.5, 2.5, 2.0),
            ac_power=2.5,
            window_area=4.0,
            insulation_factor=1.0
        )
    else:
        config = load_model_config(args.config)
        room_params = RoomParams(**config['room_params'])
        logger.info(f"Loaded configuration from: {args.config}")
    
    # Create external conditions
    conditions = ExternalConditions(
        outdoor_temp=args.outdoor_temp,
        solar_radiation=args.solar_radiation,
        wind_speed=args.wind_speed,
        time_of_day=args.time
    )
    
    logger.info(f"\nExternal Conditions:")
    logger.info(f"  Outdoor temperature: {conditions.outdoor_temp}°C")
    logger.info(f"  Solar radiation: {conditions.solar_radiation}W/m²")
    logger.info(f"  Wind speed: {conditions.wind_speed}m/s")
    logger.info(f"  Time: {conditions.time_of_day}:00")
    
    # Load model
    logger.info(f"\nLoading model from: {args.model}")
    
    try:
        # Try to load with NPU adapter
        adapter = NPUAdapter(backend=args.backend)
        adapter.load_model(args.model)
        logger.info(f"Model loaded with {args.backend} backend")
        
        # Run inference
        logger.info("\nRunning inference...")
        
        # Generate sample points for prediction
        room_size = room_params.room_volume ** (1/3)
        
        # Predict at center of room
        center_point = np.array([[
            room_size / 2,  # x
            room_size / 2,  # y
            1.5,            # z (seated height)
            args.time * 3600  # t (convert to seconds)
        ]])
        
        prediction = adapter.predict(center_point)
        
        logger.info(f"\nPrediction Results:")
        logger.info(f"  Location: Center of room (x={room_size/2:.1f}, y={room_size/2:.1f}, z=1.5m)")
        logger.info(f"  Predicted temperature: {prediction[0, 0]:.2f}°C")
        
        # Predict temperature distribution
        logger.info("\nTemperature Distribution (simplified):")
        
        # Sample at different heights
        heights = [0.5, 1.0, 1.5, 2.0, 2.5]
        for h in heights:
            point = np.array([[room_size/2, room_size/2, h, args.time * 3600]])
            temp = adapter.predict(point)
            logger.info(f"  Height {h}m: {temp[0, 0]:.2f}°C")
        
        # Benchmark if requested
        if args.benchmark:
            logger.info("\nRunning performance benchmark...")
            benchmark_results = adapter.benchmark(n_iterations=100, batch_size=1)
            
            logger.info("\nBenchmark Results:")
            logger.info(f"  Mean latency: {benchmark_results['mean_latency_ms']:.2f}ms")
            logger.info(f"  P95 latency: {benchmark_results['p95_latency_ms']:.2f}ms")
            logger.info(f"  P99 latency: {benchmark_results['p99_latency_ms']:.2f}ms")
            logger.info(f"  Throughput: {benchmark_results['throughput_qps']:.2f} QPS")
            
            # Check if meets requirements
            if benchmark_results['mean_latency_ms'] < 100:
                logger.info("  ✓ Meets real-time requirement (<100ms)")
            else:
                logger.warning("  ✗ Does not meet real-time requirement")
        
        # Visualization
        if args.visualize:
            try:
                import matplotlib.pyplot as plt
                
                logger.info("\nGenerating visualization...")
                
                # Create temperature field
                nx, ny = 20, 20
                x = np.linspace(0, room_size, nx)
                y = np.linspace(0, room_size, ny)
                X, Y = np.meshgrid(x, y)
                
                # Predict at each point (at z=1.5m, fixed time)
                T = np.zeros((nx, ny))
                for i in range(nx):
                    for j in range(ny):
                        point = np.array([[X[i, j], Y[i, j], 1.5, args.time * 3600]])
                        T[i, j] = adapter.predict(point)[0, 0]
                
                # Plot
                plt.figure(figsize=(10, 8))
                plt.contourf(X, Y, T, levels=20, cmap='RdYlBu_r')
                plt.colorbar(label='Temperature (°C)')
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
                plt.title(f'Temperature Distribution at t={args.time}h, z=1.5m')
                
                # Mark AC position
                ac_x, ac_y, _ = room_params.ac_position
                plt.plot(ac_x, ac_y, 'b*', markersize=15, label='AC Unit')
                plt.legend()
                
                output_path = 'temperature_distribution.png'
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Visualization saved to: {output_path}")
                
            except ImportError:
                logger.warning("Matplotlib not available, skipping visualization")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def main():
    """Main function."""
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        logger.info("Please train a model first using train_example.py")
        return
    
    run_inference(args)
    
    logger.info("\n" + "=" * 60)
    logger.info("Inference Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
