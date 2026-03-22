"""
PINN (Physics-Informed Neural Network) module for building thermal modeling.

This module provides physics-informed neural network implementations for
predicting temperature distribution in buildings based on physical parameters
and external conditions.
"""

from .thermal_pinn import ThermalPINN, RoomParams, ExternalConditions
from .data_generator import CFDDataGenerator
from .export import ONNXExporter

__version__ = "1.0.0"
__all__ = [
    "ThermalPINN",
    "RoomParams", 
    "ExternalConditions",
    "CFDDataGenerator",
    "ONNXExporter",
]
