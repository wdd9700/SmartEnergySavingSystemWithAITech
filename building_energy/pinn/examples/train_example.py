"""
PINN Training Example

This script demonstrates how to train a ThermalPINN model for building
temperature prediction.

Usage:
    python train_example.py --epochs 10000 --output models/pinn
"""

import os
import sys
import argparse
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from building_energy.pinn import ThermalPINN, RoomParams, ExternalConditions
from building_energy.pinn.trainer import PINNTrainer, TrainingConfig, train_pinn_model
from building_energy.pinn.data_generator import CFDDataGenerator, SimulationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ThermalPINN model')
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10000,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        default='pytorch',
        choices=['pytorch', 'mindspore', 'paddle', 'tensorflow'],
        help='Computation backend'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models/pinn',
        help='Output directory for models'
    )
    
    parser.add_argument(
        '--wall-thickness',
        type=float,
        default=0.2,
        help='Wall thickness in meters'
    )
    
    parser.add_argument(
        '--room-volume',
        type=float,
        default=50.0,
        help='Room volume in cubic meters'
    )
    
    parser.add_argument(
        '--ac-power',
        type=float,
        default=2.5,
        help='AC power in kW'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with reduced epochs'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Adjust for quick test
    if args.quick_test:
        args.epochs = 100
        logger.info("Quick test mode: using 100 epochs")
    
    logger.info("=" * 60)
    logger.info("ThermalPINN Training Example")
    logger.info("=" * 60)
    
    # Create room parameters
    room_params = RoomParams(
        wall_thickness=args.wall_thickness,
        room_volume=args.room_volume,
        ac_position=(args.room_volume**(1/3)/2, args.room_volume**(1/3)/2, 2.0),
        ac_power=args.ac_power,
        window_area=4.0,
        insulation_factor=1.0
    )
    
    logger.info(f"Room Parameters:")
    logger.info(f"  Wall thickness: {room_params.wall_thickness}m")
    logger.info(f"  Room volume: {room_params.room_volume}m³")
    logger.info(f"  AC power: {room_params.ac_power}kW")
    logger.info(f"  Window area: {room_params.window_area}m²")
    
    # Create training configuration
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        n_training_samples=5000 if not args.quick_test else 500,
        n_validation_samples=1000 if not args.quick_test else 100
    )
    
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Epochs: {training_config.epochs}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")
    logger.info(f"  Training samples: {training_config.n_training_samples}")
    logger.info(f"  Backend: {args.backend}")
    
    # Train model
    logger.info("\nStarting training...")
    
    try:
        trainer = train_pinn_model(
            room_params=room_params,
            training_config=training_config,
            output_dir=args.output,
            backend=args.backend
        )
        
        # Print training report
        report = trainer.get_training_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Final loss: {report['training_history']['final_loss']:.6f}")
        logger.info(f"Training time: {report['training_history']['training_time']:.2f}s")
        logger.info(f"Model saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
