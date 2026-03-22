"""
CFD Data Generator for PINN Training

This module generates synthetic training data for the ThermalPINN model
using simplified CFD (Computational Fluid Dynamics) simulations.
The data includes temperature distributions based on room parameters
and external conditions.
"""

import logging
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np

from .thermal_pinn import RoomParams, ExternalConditions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for CFD simulation."""
    nx: int = 20  # Grid points in x
    ny: int = 20  # Grid points in y
    nz: int = 10  # Grid points in z
    dt: float = 60.0  # Time step in seconds
    n_hours: int = 24  # Simulation duration in hours
    
    def validate(self):
        """Validate simulation configuration."""
        assert self.nx > 0 and self.ny > 0 and self.nz > 0, "Grid dimensions must be positive"
        assert self.dt > 0, "Time step must be positive"
        assert self.n_hours > 0, "Simulation duration must be positive"


class CFDDataGenerator:
    """
    Simplified CFD simulator for generating PINN training data.
    
    This class implements a finite difference solver for the heat equation
    to generate synthetic temperature data. It's designed to be fast and
    produce physically plausible data, not to be a high-fidelity CFD solver.
    
    Attributes:
        room_params: Physical room parameters
        config: Simulation configuration
        grid: Spatial grid coordinates
        temperature: Current temperature field
    """
    
    def __init__(
        self,
        room_params: RoomParams,
        config: Optional[SimulationConfig] = None
    ):
        """
        Initialize the CFD data generator.
        
        Args:
            room_params: Physical room parameters
            config: Simulation configuration
        """
        room_params.validate()
        self.room_params = room_params
        
        self.config = config or SimulationConfig()
        self.config.validate()
        
        # Compute room dimensions from volume (assume cubic room)
        self.room_size = room_params.room_volume ** (1/3)
        self.height = 3.0  # Standard ceiling height
        
        # Create spatial grid
        self._setup_grid()
        
        # Initialize temperature field
        self.temperature = None
        self.time_history = []
        self.temp_history = []
        
        # Compute thermal properties
        self._compute_thermal_properties()
        
        logger.info(f"CFDDataGenerator initialized: {self.config.nx}x{self.config.ny}x{self.config.nz} grid")
    
    def _setup_grid(self):
        """Set up the spatial grid."""
        self.x = np.linspace(0, self.room_size, self.config.nx)
        self.y = np.linspace(0, self.room_size, self.config.ny)
        self.z = np.linspace(0, self.height, self.config.nz)
        
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        
        # Create meshgrid
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
    
    def _compute_thermal_properties(self):
        """Compute thermal properties based on room parameters."""
        # Thermal diffusivity (m²/s)
        base_alpha = 1e-5
        insulation_effect = 1.0 / self.room_params.insulation_factor
        thickness_effect = 0.2 / self.room_params.wall_thickness
        self.alpha = base_alpha * insulation_effect * thickness_effect
        
        # Heat transfer coefficients
        self.h_wall = 0.5 / self.room_params.wall_thickness  # Wall heat transfer
        self.h_window = 5.8 * self.room_params.window_area / 10.0  # Window heat transfer
        
        logger.info(f"Thermal diffusivity: {self.alpha:.2e} m²/s")
    
    def _initialize_temperature(self, initial_temp: float = 25.0):
        """Initialize temperature field."""
        self.temperature = np.ones((self.config.nx, self.config.ny, self.config.nz)) * initial_temp
        self.time_history = [0.0]
        self.temp_history = [self.temperature.copy()]
    
    def _apply_boundary_conditions(self, outdoor_temp: float, solar_radiation: float):
        """
        Apply boundary conditions to temperature field.
        
        Args:
            outdoor_temp: Outdoor temperature (°C)
            solar_radiation: Solar radiation (W/m²)
        """
        T = self.temperature
        
        # Wall boundaries (x=0, x=L, y=0, y=L)
        # Robin boundary condition: -k*dT/dn = h*(T - T_outdoor)
        
        # x=0 wall (with windows)
        window_effect = solar_radiation / 1000.0 * 0.3  # Solar gain through windows
        T[0, :, :] = (T[1, :, :] + self.h_wall * self.dx * (outdoor_temp + window_effect)) / (1 + self.h_wall * self.dx)
        
        # x=L wall
        T[-1, :, :] = (T[-2, :, :] + self.h_wall * self.dx * outdoor_temp) / (1 + self.h_wall * self.dx)
        
        # y=0 wall
        T[:, 0, :] = (T[:, 1, :] + self.h_wall * self.dy * outdoor_temp) / (1 + self.h_wall * self.dy)
        
        # y=L wall
        T[:, -1, :] = (T[:, -2, :] + self.h_wall * self.dy * outdoor_temp) / (1 + self.h_wall * self.dy)
        
        # Floor (z=0) and ceiling (z=H)
        T[:, :, 0] = (T[:, :, 1] + self.h_wall * self.dz * outdoor_temp) / (1 + self.h_wall * self.dz)
        T[:, :, -1] = (T[:, :, -2] + self.h_wall * self.dz * outdoor_temp) / (1 + self.h_wall * self.dz)
    
    def _add_ac_effect(self):
        """Add air conditioning heating/cooling effect."""
        ac_x, ac_y, ac_z = self.room_params.ac_position
        
        # Find nearest grid point to AC position
        ix = np.argmin(np.abs(self.x - ac_x))
        iy = np.argmin(np.abs(self.y - ac_y))
        iz = np.argmin(np.abs(self.z - ac_z))
        
        # AC effect radius (in grid points)
        radius = 3
        
        # Add cooling/heating effect
        for i in range(max(0, ix-radius), min(self.config.nx, ix+radius+1)):
            for j in range(max(0, iy-radius), min(self.config.ny, iy+radius+1)):
                for k in range(max(0, iz-radius), min(self.config.nz, iz+radius+1)):
                    dist = np.sqrt((i-ix)**2 + (j-iy)**2 + (k-iz)**2)
                    if dist <= radius:
                        # Cooling effect (negative for AC)
                        effect = -self.room_params.ac_power * 0.5 * np.exp(-dist / 2.0)
                        self.temperature[i, j, k] += effect
    
    def _compute_laplacian(self) -> np.ndarray:
        """Compute Laplacian of temperature field."""
        T = self.temperature
        laplacian = np.zeros_like(T)
        
        # Interior points
        laplacian[1:-1, 1:-1, 1:-1] = (
            (T[2:, 1:-1, 1:-1] - 2*T[1:-1, 1:-1, 1:-1] + T[:-2, 1:-1, 1:-1]) / self.dx**2 +
            (T[1:-1, 2:, 1:-1] - 2*T[1:-1, 1:-1, 1:-1] + T[1:-1, :-2, 1:-1]) / self.dy**2 +
            (T[1:-1, 1:-1, 2:] - 2*T[1:-1, 1:-1, 1:-1] + T[1:-1, 1:-1, :-2]) / self.dz**2
        )
        
        return laplacian
    
    def _time_step(self):
        """Perform one time step of the heat equation."""
        # Compute Laplacian
        laplacian = self._compute_laplacian()
        
        # Update temperature: T_new = T_old + alpha * dt * laplacian(T)
        self.temperature[1:-1, 1:-1, 1:-1] += (
            self.alpha * self.config.dt * laplacian[1:-1, 1:-1, 1:-1]
        )
        
        # Add AC effect
        self._add_ac_effect()
    
    def run_simulation(
        self,
        conditions: ExternalConditions,
        initial_temp: float = 25.0,
        save_interval: int = 60  # Save every N time steps
    ) -> Dict[str, Any]:
        """
        Run the CFD simulation.
        
        Args:
            conditions: External environmental conditions
            initial_temp: Initial temperature (°C)
            save_interval: Save data every N time steps
            
        Returns:
            Dictionary containing simulation results
        """
        conditions.validate()
        
        # Initialize
        self._initialize_temperature(initial_temp)
        
        n_steps = int(self.config.n_hours * 3600 / self.config.dt)
        logger.info(f"Running simulation for {self.config.n_hours} hours ({n_steps} steps)")
        
        # Time stepping
        for step in range(n_steps):
            current_time = step * self.config.dt
            
            # Apply boundary conditions
            self._apply_boundary_conditions(
                conditions.outdoor_temp,
                conditions.solar_radiation
            )
            
            # Time step
            self._time_step()
            
            # Save data at intervals
            if step % save_interval == 0:
                self.time_history.append(current_time)
                self.temp_history.append(self.temperature.copy())
                
                # Log progress
                if step % (save_interval * 10) == 0:
                    avg_temp = np.mean(self.temperature)
                    logger.debug(f"Step {step}/{n_steps}, Time: {current_time/3600:.1f}h, Avg temp: {avg_temp:.2f}°C")
        
        logger.info(f"Simulation completed. Generated {len(self.time_history)} snapshots")
        
        return {
            "time": np.array(self.time_history),
            "temperature": np.array(self.temp_history),
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "room_params": self.room_params,
            "conditions": conditions
        }
    
    def generate_training_data(
        self,
        n_samples: int = 1000,
        conditions_list: Optional[List[ExternalConditions]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate training data for PINN.
        
        Args:
            n_samples: Number of training samples to generate
            conditions_list: List of external conditions (random if None)
            
        Returns:
            Dictionary with training data arrays
        """
        logger.info(f"Generating {n_samples} training samples...")
        
        if conditions_list is None:
            # Generate random conditions
            conditions_list = self._generate_random_conditions(n_samples)
        
        all_data = {
            "x": [],
            "y": [],
            "z": [],
            "t": [],
            "T": [],
            "outdoor_temp": [],
            "solar_radiation": [],
            "wall_thickness": [],
            "room_volume": [],
            "ac_power": []
        }
        
        samples_per_condition = max(1, n_samples // len(conditions_list))
        
        for conditions in conditions_list:
            # Run simulation
            results = self.run_simulation(conditions)
            
            # Extract samples from simulation results
            n_time_steps = len(results["time"])
            sample_indices = np.random.choice(
                n_time_steps,
                min(samples_per_condition, n_time_steps),
                replace=False
            )
            
            for idx in sample_indices:
                # Random spatial points
                n_points = 10
                for _ in range(n_points):
                    ix = np.random.randint(0, self.config.nx)
                    iy = np.random.randint(0, self.config.ny)
                    iz = np.random.randint(0, self.config.nz)
                    
                    all_data["x"].append(self.x[ix])
                    all_data["y"].append(self.y[iy])
                    all_data["z"].append(self.z[iz])
                    all_data["t"].append(results["time"][idx])
                    all_data["T"].append(results["temperature"][idx, ix, iy, iz])
                    all_data["outdoor_temp"].append(conditions.outdoor_temp)
                    all_data["solar_radiation"].append(conditions.solar_radiation)
                    all_data["wall_thickness"].append(self.room_params.wall_thickness)
                    all_data["room_volume"].append(self.room_params.room_volume)
                    all_data["ac_power"].append(self.room_params.ac_power)
        
        # Convert to numpy arrays
        for key in all_data:
            all_data[key] = np.array(all_data[key])
        
        # Validate temperature range
        temp_min, temp_max = np.min(all_data["T"]), np.max(all_data["T"])
        assert -20 <= temp_min <= 50, f"Temperature {temp_min}°C out of valid range"
        assert -20 <= temp_max <= 50, f"Temperature {temp_max}°C out of valid range"
        
        logger.info(f"Generated {len(all_data['T'])} training points")
        logger.info(f"Temperature range: {temp_min:.2f} - {temp_max:.2f}°C")
        
        return all_data
    
    def _generate_random_conditions(self, n_samples: int) -> List[ExternalConditions]:
        """Generate random external conditions for training."""
        n_conditions = min(n_samples // 100, 20)  # Up to 20 different conditions
        
        conditions_list = []
        for _ in range(n_conditions):
            conditions = ExternalConditions(
                outdoor_temp=np.random.uniform(-10, 40),
                solar_radiation=np.random.uniform(0, 1000),
                wind_speed=np.random.uniform(0, 20),
                time_of_day=np.random.uniform(0, 24)
            )
            conditions_list.append(conditions)
        
        return conditions_list
    
    def validate_data(self, data: Dict[str, np.ndarray]) -> bool:
        """
        Validate generated training data.
        
        Args:
            data: Training data dictionary
            
        Returns:
            True if data is valid
        """
        # Check temperature range
        temps = data["T"]
        if np.any(temps < -20) or np.any(temps > 50):
            logger.error("Temperature out of valid range [-20, 50]°C")
            return False
        
        # Check for NaN or Inf
        for key, values in data.items():
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                logger.error(f"NaN or Inf values found in {key}")
                return False
        
        # Check spatial coordinates are within room
        room_size = self.room_params.room_volume ** (1/3)
        if np.any(data["x"] < 0) or np.any(data["x"] > room_size):
            logger.error("X coordinates out of bounds")
            return False
        if np.any(data["y"] < 0) or np.any(data["y"] > room_size):
            logger.error("Y coordinates out of bounds")
            return False
        if np.any(data["z"] < 0) or np.any(data["z"] > self.height):
            logger.error("Z coordinates out of bounds")
            return False
        
        logger.info("Data validation passed")
        return True


def generate_parameter_sweep(
    n_configs: int = 10,
    n_samples_per_config: int = 100
) -> List[Dict[str, Any]]:
    """
    Generate training data across a range of room parameters.
    
    Args:
        n_configs: Number of different room configurations
        n_samples_per_config: Samples per configuration
        
    Returns:
        List of training data dictionaries
    """
    logger.info(f"Generating parameter sweep: {n_configs} configs x {n_samples_per_config} samples")
    
    all_data = []
    
    for i in range(n_configs):
        # Random room parameters
        room_params = RoomParams(
            wall_thickness=np.random.uniform(0.1, 0.5),
            room_volume=np.random.uniform(20, 200),
            ac_position=(
                np.random.uniform(1, 5),
                np.random.uniform(1, 5),
                np.random.uniform(1.5, 2.5)
            ),
            ac_power=np.random.uniform(1, 5),
            window_area=np.random.uniform(1, 10),
            insulation_factor=np.random.uniform(0.5, 2.0)
        )
        
        # Create generator
        generator = CFDDataGenerator(room_params)
        
        # Generate data
        data = generator.generate_training_data(n_samples=n_samples_per_config)
        
        if generator.validate_data(data):
            all_data.append({
                "room_params": room_params,
                "data": data
            })
        else:
            logger.warning(f"Config {i} failed validation, skipping")
    
    logger.info(f"Parameter sweep complete: {len(all_data)} valid configurations")
    return all_data
