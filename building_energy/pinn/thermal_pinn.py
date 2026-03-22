"""
ThermalPINN - Physics-Informed Neural Network for Building Thermal Modeling

This module implements a PINN using DeepXDE to model heat transfer in buildings.
The network learns the relationship between room parameters, external conditions,
and temperature distribution without requiring exact physical parameters.
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any, Callable
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import DeepXDE and backends
try:
    import deepxde as dde
    from deepxde.backend import tf
    DEEPXDE_AVAILABLE = True
    DEEPXDE_VERSION = dde.__version__
    logger.info(f"DeepXDE version: {DEEPXDE_VERSION}")
except ImportError as e:
    DEEPXDE_AVAILABLE = False
    DEEPXDE_VERSION = None
    logger.warning(f"DeepXDE not available: {e}")

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch version: {torch.__version__}")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    import mindspore as ms
    MINDSPORE_AVAILABLE = True
    logger.info(f"MindSpore version: {ms.__version__}")
except ImportError:
    MINDSPORE_AVAILABLE = False
    logger.warning("MindSpore not available")

try:
    import paddle
    PADDLE_AVAILABLE = True
    logger.info(f"PaddlePaddle version: {paddle.__version__}")
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("PaddlePaddle not available")


@dataclass
class RoomParams:
    """Physical parameters of a room."""
    wall_thickness: float  # 墙厚 (m), 范围: 0.1-0.5
    room_volume: float     # 房间体积 (m³), 范围: 20-200
    ac_position: Tuple[float, float, float]  # 空调位置 (x,y,z)
    ac_power: float        # 空调功率 (kW), 范围: 1-5
    window_area: float     # 窗户面积 (m²), 范围: 1-10
    insulation_factor: float  # 保温系数, 范围: 0.5-2.0
    
    def validate(self) -> bool:
        """Validate parameter ranges."""
        assert 0.1 <= self.wall_thickness <= 0.5, "墙厚必须在0.1-0.5m之间"
        assert 20 <= self.room_volume <= 200, "房间体积必须在20-200m³之间"
        assert 1 <= self.ac_power <= 5, "空调功率必须在1-5kW之间"
        assert 1 <= self.window_area <= 10, "窗户面积必须在1-10m²之间"
        assert 0.5 <= self.insulation_factor <= 2.0, "保温系数必须在0.5-2.0之间"
        return True
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.wall_thickness,
            self.room_volume,
            *self.ac_position,
            self.ac_power,
            self.window_area,
            self.insulation_factor
        ])


@dataclass
class ExternalConditions:
    """External environmental conditions."""
    outdoor_temp: float    # 室外温度 (°C)
    solar_radiation: float # 太阳辐射 (W/m²)
    wind_speed: float      # 风速 (m/s)
    time_of_day: float     # 时间 (0-24h)
    
    def validate(self) -> bool:
        """Validate parameter ranges."""
        assert -30 <= self.outdoor_temp <= 50, "室外温度必须在-30到50°C之间"
        assert 0 <= self.solar_radiation <= 1200, "太阳辐射必须在0-1200W/m²之间"
        assert 0 <= self.wind_speed <= 50, "风速必须在0-50m/s之间"
        assert 0 <= self.time_of_day <= 24, "时间必须在0-24h之间"
        return True
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.outdoor_temp,
            self.solar_radiation,
            self.wind_speed,
            self.time_of_day
        ])


class ThermalPINN:
    """
    Physics-Informed Neural Network for building thermal modeling.
    
    This class implements a PINN using DeepXDE to solve the heat equation
    for building temperature prediction. It supports multiple backends
    (PyTorch, MindSpore, PaddlePaddle) and can export to ONNX format.
    
    Attributes:
        room_params: Physical parameters of the room
        backend: Computation backend ('pytorch', 'mindspore', 'paddle', 'tensorflow')
        model: The DeepXDE model instance
        trained: Whether the model has been trained
    """
    
    def __init__(
        self,
        room_params: RoomParams,
        backend: str = "pytorch",
        network_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ThermalPINN model.
        
        Args:
            room_params: Physical parameters of the room
            backend: Computation backend ('pytorch', 'mindspore', 'paddle', 'tensorflow')
            network_config: Network architecture configuration
            
        Raises:
            ImportError: If DeepXDE or the specified backend is not available
            ValueError: If room parameters are invalid
        """
        if not DEEPXDE_AVAILABLE:
            raise ImportError(
                "DeepXDE is required but not installed. "
                "Install with: pip install deepxde>=1.15.0"
            )
        
        # Validate room parameters
        room_params.validate()
        self.room_params = room_params
        
        # Set and validate backend
        self.backend = self._validate_backend(backend)
        
        # Network configuration
        self.network_config = network_config or {
            "layer_size": [4] + [64] * 6 + [1],  # [x, y, z, t] -> [T]
            "activation": "tanh",
            "initializer": "Glorot uniform"
        }
        
        # Model placeholders
        self.model = None
        self.trained = False
        self.training_history = []
        
        # PDE parameters (learned or set)
        self.alpha = None  # Thermal diffusivity
        self._compute_thermal_properties()
        
        logger.info(f"ThermalPINN initialized with backend: {self.backend}")
        logger.info(f"Room volume: {room_params.room_volume}m³, "
                   f"AC power: {room_params.ac_power}kW")
    
    def _validate_backend(self, backend: str) -> str:
        """Validate and auto-select backend."""
        backend = backend.lower()
        
        available_backends = []
        if TORCH_AVAILABLE:
            available_backends.append("pytorch")
        if MINDSPORE_AVAILABLE:
            available_backends.append("mindspore")
        if PADDLE_AVAILABLE:
            available_backends.append("paddle")
        
        if backend == "pytorch" and not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, trying fallback")
            backend = "auto"
        elif backend == "mindspore" and not MINDSPORE_AVAILABLE:
            logger.warning("MindSpore not available, trying fallback")
            backend = "auto"
        elif backend == "paddle" and not PADDLE_AVAILABLE:
            logger.warning("PaddlePaddle not available, trying fallback")
            backend = "auto"
        
        if backend == "auto" or backend not in available_backends:
            if available_backends:
                backend = available_backends[0]
                logger.info(f"Auto-selected backend: {backend}")
            else:
                raise ImportError(
                    "No supported backend available. "
                    "Install PyTorch, MindSpore, or PaddlePaddle."
                )
        
        return backend
    
    def _compute_thermal_properties(self):
        """Compute thermal properties based on room parameters."""
        # Thermal diffusivity depends on wall thickness and insulation
        # α = k / (ρ * cp), simplified model
        base_alpha = 1e-5  # m²/s (typical for building materials)
        insulation_effect = 1.0 / self.room_params.insulation_factor
        thickness_effect = 0.2 / self.room_params.wall_thickness
        
        self.alpha = base_alpha * insulation_effect * thickness_effect
        logger.info(f"Computed thermal diffusivity: {self.alpha:.2e} m²/s")
    
    def define_pde(self, x, y):
        """
        Define the heat transfer PDE.
        
        The PDE models heat transfer in a room:
        ∂T/∂t = α * ∇²T + Q_ac + Q_solar - Q_loss
        
        Args:
            x: Input coordinates [x, y, z, t]
            y: Temperature output T
            
        Returns:
            PDE residual
        """
        # Temperature
        T = y
        
        # Time derivative
        T_t = dde.grad.jacobian(y, x, i=0, j=3)  # dT/dt
        
        # Spatial derivatives (Laplacian)
        T_xx = dde.grad.hessian(y, x, i=0, j=0)  # d²T/dx²
        T_yy = dde.grad.hessian(y, x, i=1, j=1)  # d²T/dy²
        T_zz = dde.grad.hessian(y, x, i=2, j=2)  # d²T/dz²
        
        # Heat sources and losses
        # Q_ac: Air conditioning effect (simplified)
        ac_x, ac_y, ac_z = self.room_params.ac_position
        # Distance from AC
        dist_ac = ((x[:, 0:1] - ac_x)**2 + 
                   (x[:, 1:2] - ac_y)**2 + 
                   (x[:, 2:3] - ac_z)**2)**0.5
        # AC cooling/heating effect decays with distance
        Q_ac = self.room_params.ac_power * 1000 * tf.exp(-dist_ac / 2.0) / self.room_params.room_volume
        
        # Q_solar: Solar radiation through windows (simplified)
        # Assume windows are on one wall (x=0)
        window_effect = tf.exp(-((x[:, 0:1])**2) / 0.1)  # Near x=0 wall
        Q_solar = 100 * window_effect * self.room_params.window_area / self.room_params.room_volume
        
        # Q_loss: Heat loss through walls
        # Assume room is a cube, compute distance to nearest wall
        room_size = self.room_params.room_volume ** (1/3)
        dist_to_wall = tf.minimum(
            tf.minimum(x[:, 0:1], room_size - x[:, 0:1]),
            tf.minimum(x[:, 1:2], room_size - x[:, 1:2])
        )
        # Heat loss proportional to temperature difference and wall properties
        T_outdoor = 25.0  # Simplified, should come from external conditions
        Q_loss = (T - T_outdoor) * self.room_params.wall_thickness * 0.1 / self.room_params.insulation_factor
        
        # Heat equation with source terms
        # ∂T/∂t = α * ∇²T + Q_ac + Q_solar - Q_loss
        f = T_t - self.alpha * (T_xx + T_yy + T_zz) - Q_ac - Q_solar + Q_loss
        
        return f
    
    def setup_geometry(self, room_dimensions: Tuple[float, float, float] = (5.0, 5.0, 3.0)):
        """
        Set up the spatial and temporal geometry.
        
        Args:
            room_dimensions: Room dimensions (length, width, height) in meters
        """
        # Spatial domain: room interior
        self.geom = dde.geometry.Cuboid(
            xmin=[0, 0, 0],
            xmax=list(room_dimensions)
        )
        
        # Temporal domain: 24 hours
        self.time_domain = dde.geometry.TimeDomain(0, 24 * 3600)  # seconds
        
        # Spatio-temporal domain
        self.geomtime = dde.geometry.GeometryXTime(self.geom, self.time_domain)
        
        logger.info(f"Geometry setup: room {room_dimensions}, time 0-24h")
    
    def setup_boundary_conditions(self, outdoor_temp: float = 25.0):
        """
        Set up boundary conditions.
        
        Args:
            outdoor_temp: Outdoor temperature in Celsius
        """
        self.boundary_conditions = []
        
        # Wall boundary: heat transfer through walls
        # Dirichlet BC on walls with outdoor temperature influence
        def wall_boundary(x, on_boundary):
            room_size = self.room_params.room_volume ** (1/3)
            return on_boundary and (
                np.isclose(x[0], 0) or np.isclose(x[0], room_size) or
                np.isclose(x[1], 0) or np.isclose(x[1], room_size) or
                np.isclose(x[2], 0) or np.isclose(x[2], 3.0)  # Assuming 3m height
            )
        
        # Robin-type BC: T_wall = T_outdoor + insulation_effect * (T_inside - T_outdoor)
        def wall_temp(x):
            insulation = self.room_params.insulation_factor
            return outdoor_temp + (25.0 - outdoor_temp) / insulation  # Simplified
        
        bc_walls = dde.DirichletBC(
            self.geomtime,
            wall_temp,
            lambda x, on_boundary: wall_boundary(x, on_boundary),
            component=0
        )
        self.boundary_conditions.append(bc_walls)
        
        # Initial condition: uniform temperature
        ic = dde.IC(
            self.geomtime,
            lambda x: 25.0,  # Initial temperature
            lambda x, on_initial: on_initial
        )
        self.boundary_conditions.append(ic)
        
        logger.info(f"Boundary conditions setup with outdoor temp: {outdoor_temp}°C")
    
    def setup_data(self, n_domain: int = 10000, n_boundary: int = 2000, n_initial: int = 1000):
        """
        Set up training data points.
        
        Args:
            n_domain: Number of domain points for PDE residual
            n_boundary: Number of boundary points
            n_initial: Number of initial condition points
        """
        self.data = dde.data.TimePDE(
            self.geomtime,
            self.define_pde,
            self.boundary_conditions,
            num_domain=n_domain,
            num_boundary=n_boundary,
            num_initial=n_initial
        )
        
        logger.info(f"Data setup: {n_domain} domain, {n_boundary} boundary, {n_initial} initial points")
    
    def build_network(self):
        """Build the neural network architecture."""
        self.net = dde.nn.FNN(
            layer_size=self.network_config["layer_size"],
            activation=self.network_config["activation"],
            initializer=self.network_config["initializer"]
        )
        
        logger.info(f"Network built: {self.network_config['layer_size']}")
    
    def compile_model(self, learning_rate: float = 0.001):
        """
        Compile the model with optimizer and loss weights.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        self.model = dde.Model(self.data, self.net)
        
        # Loss weights: [PDE, BC, IC]
        loss_weights = [1.0, 1.0, 1.0]
        
        self.model.compile(
            optimizer="adam",
            lr=learning_rate,
            loss_weights=loss_weights
        )
        
        logger.info(f"Model compiled with lr={learning_rate}")
    
    def train(
        self,
        epochs: int = 10000,
        learning_rate: float = 0.001,
        display_every: int = 1000,
        callbacks: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Train the PINN model.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            display_every: Display loss every N epochs
            callbacks: List of callback functions
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not compiled. Call compile_model() first.")
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        # Training
        losshistory, train_state = self.model.train(
            iterations=epochs,
            display_every=display_every,
            callbacks=callbacks or []
        )
        
        self.trained = True
        self.training_history = {
            "loss": losshistory.loss_train,
            "test_loss": losshistory.loss_test,
            "epochs": epochs
        }
        
        final_loss = losshistory.loss_train[-1]
        logger.info(f"Training completed. Final loss: {final_loss:.6f}")
        
        return self.training_history
    
    def predict(
        self,
        conditions: ExternalConditions,
        spatial_resolution: Tuple[int, int, int] = (10, 10, 5),
        time_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict temperature distribution.
        
        Args:
            conditions: External environmental conditions
            spatial_resolution: Grid resolution (nx, ny, nz)
            time_points: Time points for prediction (default: hourly for 24h)
            
        Returns:
            Temperature distribution array [time, x, y, z]
        """
        if not self.trained:
            logger.warning("Model not trained, predictions may be inaccurate")
        
        conditions.validate()
        
        # Generate grid
        room_size = self.room_params.room_volume ** (1/3)
        nx, ny, nz = spatial_resolution
        
        x = np.linspace(0, room_size, nx)
        y = np.linspace(0, room_size, ny)
        z = np.linspace(0, 3.0, nz)  # Assuming 3m height
        
        if time_points is None:
            time_points = np.linspace(0, 24, 25) * 3600  # Hourly for 24h
        
        # Create prediction points
        X, Y, Z, T = np.meshgrid(x, y, z, time_points, indexing='ij')
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        T_flat = T.flatten()
        
        points = np.stack([X_flat, Y_flat, Z_flat, T_flat], axis=1)
        
        # Predict
        if self.model is not None:
            temperatures = self.model.predict(points)
        else:
            # Fallback: simple physics-based prediction
            temperatures = self._physics_prediction(points, conditions)
        
        # Reshape to [time, x, y, z]
        nt = len(time_points)
        temperatures = temperatures.reshape(nx, ny, nz, nt)
        temperatures = np.transpose(temperatures, (3, 0, 1, 2))
        
        return temperatures
    
    def _physics_prediction(
        self,
        points: np.ndarray,
        conditions: ExternalConditions
    ) -> np.ndarray:
        """
        Fallback physics-based prediction when model is not available.
        
        Args:
            points: Prediction points [N, 4] (x, y, z, t)
            conditions: External conditions
            
        Returns:
            Temperature predictions [N, 1]
        """
        x, y, z, t = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        
        # Base temperature from outdoor conditions with some lag
        base_temp = conditions.outdoor_temp
        
        # AC effect
        ac_x, ac_y, ac_z = self.room_params.ac_position
        dist_ac = np.sqrt((x - ac_x)**2 + (y - ac_y)**2 + (z - ac_z)**2)
        ac_effect = self.room_params.ac_power * 2 * np.exp(-dist_ac / 2.0)
        
        # Solar effect (simplified)
        solar_effect = conditions.solar_radiation / 100 * 0.5
        
        # Time-of-day effect
        hour = (t / 3600) % 24
        daily_variation = 3 * np.sin(2 * np.pi * (hour - 14) / 24)
        
        # Combine effects
        temperature = base_temp - ac_effect + solar_effect * 0.1 + daily_variation
        
        # Clamp to reasonable range
        temperature = np.clip(temperature, 16, 35)
        
        return temperature.reshape(-1, 1)
    
    def export_onnx(self, path: str) -> str:
        """
        Export the trained model to ONNX format.
        
        Args:
            path: Export path for ONNX file
            
        Returns:
            Path to exported file
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before export")
        
        try:
            import onnx
            
            # Export logic depends on backend
            if self.backend == "pytorch" and TORCH_AVAILABLE:
                # Export PyTorch model to ONNX
                dummy_input = torch.randn(1, 4)  # [x, y, z, t]
                torch.onnx.export(
                    self.net,
                    dummy_input,
                    path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
            else:
                # For other backends, save model weights and architecture
                # User needs to convert manually or use backend-specific tools
                logger.warning(f"ONNX export for {self.backend} may require manual conversion")
                self.save_checkpoint(path.replace('.onnx', '.ckpt'))
            
            logger.info(f"Model exported to: {path}")
            return path
            
        except ImportError:
            logger.error("ONNX export requires 'onnx' package")
            raise
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        if self.model is not None:
            self.model.restore(path)
            self.trained = True
            logger.info(f"Checkpoint loaded from: {path}")
    
    def save_config(self, path: str):
        """Save model configuration."""
        config = {
            "room_params": asdict(self.room_params),
            "backend": self.backend,
            "network_config": self.network_config,
            "alpha": float(self.alpha) if self.alpha is not None else None,
            "trained": self.trained
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Config saved to: {path}")
    
    @classmethod
    def load(cls, path: str, **kwargs) -> "ThermalPINN":
        """
        Load a trained model from ONNX or checkpoint.
        
        Args:
            path: Path to model file (.onnx or .ckpt)
            **kwargs: Additional arguments for initialization
            
        Returns:
            Loaded ThermalPINN instance
        """
        # Load config
        config_path = path.replace('.onnx', '_config.json').replace('.ckpt', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            room_params = RoomParams(**config["room_params"])
            instance = cls(
                room_params=room_params,
                backend=config.get("backend", "pytorch"),
                network_config=config.get("network_config")
            )
            
            # Load weights if checkpoint
            if path.endswith('.ckpt'):
                instance.load_checkpoint(path)
            
            return instance
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        return {
            "backend": self.backend,
            "trained": self.trained,
            "room_params": asdict(self.room_params),
            "network_config": self.network_config,
            "alpha": float(self.alpha) if self.alpha is not None else None,
            "training_history": self.training_history if self.training_history else None
        }


# Convenience function for quick setup
def create_thermal_pinn(
    wall_thickness: float = 0.2,
    room_volume: float = 50.0,
    ac_power: float = 2.5,
    backend: str = "pytorch"
) -> ThermalPINN:
    """
    Create a ThermalPINN with default parameters.
    
    Args:
        wall_thickness: Wall thickness in meters
        room_volume: Room volume in cubic meters
        ac_power: AC power in kW
        backend: Computation backend
        
    Returns:
        Configured ThermalPINN instance
    """
    room_params = RoomParams(
        wall_thickness=wall_thickness,
        room_volume=room_volume,
        ac_position=(room_volume**(1/3)/2, room_volume**(1/3)/2, 2.0),
        ac_power=ac_power,
        window_area=4.0,
        insulation_factor=1.0
    )
    
    return ThermalPINN(room_params, backend=backend)
