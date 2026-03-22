"""
PINN Training Script

This module provides training functionality for the ThermalPINN model,
including data loading, model training, and checkpoint management.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
import time

import numpy as np

try:
    import deepxde as dde
    DEEPXDE_AVAILABLE = True
except ImportError:
    DEEPXDE_AVAILABLE = False

from .thermal_pinn import ThermalPINN, RoomParams, ExternalConditions
from .data_generator import CFDDataGenerator, SimulationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for PINN training."""
    # Data generation
    n_training_samples: int = 5000
    n_validation_samples: int = 1000
    
    # Training parameters
    epochs: int = 10000
    learning_rate: float = 0.001
    batch_size: int = 256
    
    # Loss weights
    pde_weight: float = 1.0
    bc_weight: float = 1.0
    ic_weight: float = 1.0
    data_weight: float = 0.1
    
    # Checkpointing
    checkpoint_interval: int = 1000
    save_best_only: bool = False
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 1000
    min_delta: float = 1e-6
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_training_samples": self.n_training_samples,
            "n_validation_samples": self.n_validation_samples,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "pde_weight": self.pde_weight,
            "bc_weight": self.bc_weight,
            "ic_weight": self.ic_weight,
            "data_weight": self.data_weight,
            "checkpoint_interval": self.checkpoint_interval,
            "save_best_only": self.save_best_only,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "min_delta": self.min_delta
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(**config_dict)


class TrainingCallback:
    """Callback for monitoring training progress."""
    
    def __init__(self, display_every: int = 100):
        self.display_every = display_every
        self.start_time = None
        self.losses = []
    
    def on_train_begin(self):
        """Called at the beginning of training."""
        self.start_time = time.time()
        logger.info("Training started")
    
    def on_epoch_end(self, epoch: int, loss: float):
        """Called at the end of each epoch."""
        self.losses.append(loss)
        
        if epoch % self.display_every == 0:
            elapsed = time.time() - self.start_time
            logger.info(f"Epoch {epoch}: loss={loss:.6f}, time={elapsed:.2f}s")
    
    def on_train_end(self):
        """Called at the end of training."""
        elapsed = time.time() - self.start_time
        logger.info(f"Training completed in {elapsed:.2f}s")


class EarlyStoppingCallback:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, patience: int = 1000, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, loss: float):
        """Check if training should stop."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.should_stop = True
            self.stopped_epoch = epoch
            logger.info(f"Early stopping triggered at epoch {epoch}")


class PINNTrainer:
    """
    Trainer for ThermalPINN models.
    
    This class handles the complete training pipeline including:
    - Data generation
    - Model setup and compilation
    - Training loop with callbacks
    - Checkpoint management
    - Validation
    
    Attributes:
        room_params: Physical room parameters
        training_config: Training configuration
        model: The ThermalPINN model
        training_data: Generated training data
    """
    
    def __init__(
        self,
        room_params: RoomParams,
        training_config: Optional[TrainingConfig] = None,
        output_dir: str = "models/pinn"
    ):
        """
        Initialize the trainer.
        
        Args:
            room_params: Physical room parameters
            training_config: Training configuration
            output_dir: Directory for saving models and logs
        """
        self.room_params = room_params
        self.training_config = training_config or TrainingConfig()
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.training_data = None
        self.validation_data = None
        
        # Training state
        self.best_loss = float('inf')
        self.training_history = []
        
        logger.info(f"PINNTrainer initialized. Output dir: {output_dir}")
    
    def generate_data(self) -> Dict[str, np.ndarray]:
        """
        Generate training and validation data.
        
        Returns:
            Dictionary containing training and validation data
        """
        logger.info("Generating training data...")
        
        # Create data generator
        sim_config = SimulationConfig(
            nx=15, ny=15, nz=8,  # Coarser grid for faster generation
            dt=120.0,  # 2-minute time steps
            n_hours=24
        )
        
        generator = CFDDataGenerator(self.room_params, sim_config)
        
        # Generate training data
        self.training_data = generator.generate_training_data(
            n_samples=self.training_config.n_training_samples
        )
        
        # Validate training data
        if not generator.validate_data(self.training_data):
            raise ValueError("Training data validation failed")
        
        # Generate validation data with different conditions
        logger.info("Generating validation data...")
        val_conditions = [
            ExternalConditions(
                outdoor_temp=30.0,
                solar_radiation=800.0,
                wind_speed=5.0,
                time_of_day=14.0
            ),
            ExternalConditions(
                outdoor_temp=5.0,
                solar_radiation=200.0,
                wind_speed=10.0,
                time_of_day=8.0
            )
        ]
        
        self.validation_data = generator.generate_training_data(
            n_samples=self.training_config.n_validation_samples,
            conditions_list=val_conditions
        )
        
        logger.info("Data generation completed")
        return self.training_data
    
    def setup_model(self, backend: str = "pytorch"):
        """
        Set up the PINN model.
        
        Args:
            backend: Computation backend
        """
        logger.info("Setting up PINN model...")
        
        # Create model
        network_config = {
            "layer_size": [4] + [64] * 6 + [1],
            "activation": "tanh",
            "initializer": "Glorot uniform"
        }
        
        self.model = ThermalPINN(
            room_params=self.room_params,
            backend=backend,
            network_config=network_config
        )
        
        # Setup geometry and boundary conditions
        room_size = self.room_params.room_volume ** (1/3)
        self.model.setup_geometry(room_dimensions=(room_size, room_size, 3.0))
        self.model.setup_boundary_conditions(outdoor_temp=25.0)
        
        # Setup data
        self.model.setup_data(
            n_domain=10000,
            n_boundary=2000,
            n_initial=1000
        )
        
        # Build network
        self.model.build_network()
        
        # Compile
        self.model.compile_model(learning_rate=self.training_config.learning_rate)
        
        logger.info("Model setup completed")
    
    def train(
        self,
        callbacks: Optional[List[Callable]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the PINN model.
        
        Args:
            callbacks: List of callback functions
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not set up. Call setup_model() first.")
        
        logger.info(f"Starting training for {self.training_config.epochs} epochs...")
        
        # Create callbacks
        if callbacks is None:
            callbacks = []
        
        progress_callback = TrainingCallback(display_every=1000)
        progress_callback.on_train_begin()
        callbacks.append(progress_callback)
        
        if self.training_config.early_stopping:
            early_stopping = EarlyStoppingCallback(
                patience=self.training_config.patience,
                min_delta=self.training_config.min_delta
            )
            callbacks.append(early_stopping)
        
        # Training loop
        start_time = time.time()
        
        history = self.model.train(
            epochs=self.training_config.epochs,
            learning_rate=self.training_config.learning_rate,
            display_every=1000,
            callbacks=callbacks
        )
        
        training_time = time.time() - start_time
        
        # Store training history
        self.training_history = {
            "epochs": len(history.get("loss", [])),
            "final_loss": history.get("loss", [])[-1] if history.get("loss") else None,
            "training_time": training_time,
            "config": self.training_config.to_dict()
        }
        
        if verbose:
            logger.info(f"Training completed in {training_time:.2f}s")
            logger.info(f"Final loss: {self.training_history['final_loss']:.6f}")
        
        return self.training_history
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the trained model on validation data.
        
        Returns:
            Dictionary with validation metrics
        """
        if not self.model or not self.model.trained:
            raise RuntimeError("Model not trained")
        
        if self.validation_data is None:
            logger.warning("No validation data available")
            return {}
        
        logger.info("Validating model...")
        
        # Make predictions on validation data
        predictions = []
        actuals = []
        
        # Sample validation points
        n_val_points = min(1000, len(self.validation_data["T"]))
        indices = np.random.choice(len(self.validation_data["T"]), n_val_points, replace=False)
        
        for idx in indices:
            x = self.validation_data["x"][idx]
            y = self.validation_data["y"][idx]
            z = self.validation_data["z"][idx]
            t = self.validation_data["t"][idx]
            
            # Create external conditions
            conditions = ExternalConditions(
                outdoor_temp=self.validation_data["outdoor_temp"][idx],
                solar_radiation=self.validation_data["solar_radiation"][idx],
                wind_speed=5.0,  # Default
                time_of_day=(t / 3600) % 24
            )
            
            # Predict (simplified - predict single point)
            # In practice, you'd predict the whole field
            pred = self.model._physics_prediction(
                np.array([[x, y, z, t]]),
                conditions
            )[0, 0]
            
            predictions.append(pred)
            actuals.append(self.validation_data["T"][idx])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Compute metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse)
        }
        
        logger.info(f"Validation metrics: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        return metrics
    
    def save_model(self, name: str = "thermal_model"):
        """
        Save the trained model and configuration.
        
        Args:
            name: Base name for saved files
        """
        if not self.model or not self.model.trained:
            raise RuntimeError("Model not trained")
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.output_dir, f"{name}.ckpt")
        self.model.save_checkpoint(checkpoint_path)
        
        # Save config
        config_path = os.path.join(self.output_dir, f"{name}_config.json")
        self.model.save_config(config_path)
        
        # Save training history
        history_path = os.path.join(self.output_dir, f"{name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Export to ONNX
        try:
            onnx_path = os.path.join(self.output_dir, f"{name}.onnx")
            self.model.export_onnx(onnx_path)
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
        
        logger.info(f"Model saved to {self.output_dir}")
    
    def load_model(self, name: str = "thermal_model"):
        """
        Load a trained model.
        
        Args:
            name: Base name for saved files
        """
        checkpoint_path = os.path.join(self.output_dir, f"{name}.ckpt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load config first
        config_path = os.path.join(self.output_dir, f"{name}_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Recreate model
        room_params = RoomParams(**config["room_params"])
        self.__init__(room_params, self.training_config, self.output_dir)
        self.setup_model(backend=config.get("backend", "pytorch"))
        
        # Load weights
        self.model.load_checkpoint(checkpoint_path)
        
        logger.info(f"Model loaded from {checkpoint_path}")
    
    def get_training_report(self) -> Dict[str, Any]:
        """
        Generate a training report.
        
        Returns:
            Dictionary containing training report
        """
        report = {
            "room_params": {
                "wall_thickness": self.room_params.wall_thickness,
                "room_volume": self.room_params.room_volume,
                "ac_power": self.room_params.ac_power,
                "window_area": self.room_params.window_area,
                "insulation_factor": self.room_params.insulation_factor
            },
            "training_config": self.training_config.to_dict(),
            "training_history": self.training_history,
            "model_info": self.model.get_model_summary() if self.model else None
        }
        
        return report


def train_pinn_model(
    room_params: Optional[RoomParams] = None,
    training_config: Optional[TrainingConfig] = None,
    output_dir: str = "models/pinn",
    backend: str = "pytorch"
) -> PINNTrainer:
    """
    Convenience function to train a PINN model.
    
    Args:
        room_params: Room parameters (default if None)
        training_config: Training configuration (default if None)
        output_dir: Output directory
        backend: Computation backend
        
    Returns:
        Trained PINNTrainer instance
    """
    # Default room parameters
    if room_params is None:
        room_params = RoomParams(
            wall_thickness=0.2,
            room_volume=50.0,
            ac_position=(2.5, 2.5, 2.0),
            ac_power=2.5,
            window_area=4.0,
            insulation_factor=1.0
        )
    
    # Create trainer
    trainer = PINNTrainer(
        room_params=room_params,
        training_config=training_config,
        output_dir=output_dir
    )
    
    # Generate data
    trainer.generate_data()
    
    # Setup model
    trainer.setup_model(backend=backend)
    
    # Train
    trainer.train()
    
    # Validate
    trainer.validate()
    
    # Save
    trainer.save_model()
    
    return trainer
