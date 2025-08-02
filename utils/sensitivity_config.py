"""
Configuration for sensitivity analysis parameters
"""
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Any

@dataclass
class SensitivityConfig:
    """Configuration class for sensitivity analysis parameters"""
    
    # Sample size configuration
    DEFAULT_SAMPLES: int = 1024  # Power of 2 for optimal Sobol performance
    MIN_SAMPLES: int = 64
    MAX_SAMPLES: int = 2048
    QUICK_MODE_SAMPLES: int = 256
    
    # Parameter ranges for Sobol analysis (as multipliers of base values)
    PARAMETER_RANGES: Dict[str, Tuple[float, float]] = None
    
    # Monte Carlo uncertainty parameters
    MC_DEFAULT_SAMPLES: int = 512
    MC_QUICK_MODE_SAMPLES: int = 128
    
    # Parameter uncertainties for Monte Carlo (as standard deviations)
    PARAMETER_UNCERTAINTIES: Dict[str, float] = None
    
    # Timeout settings
    TIMEOUT_HOURS: float = 24.0
    MAX_TIME: int = 1200  # 20 minutes
    
    # Simulation settings
    SIMULATION_DAYS: int = 365
    QUICK_MODE_DAYS: int = 180
    
    def __post_init__(self):
        """Initialize default parameter ranges if not provided"""
        if self.PARAMETER_RANGES is None:
            self.PARAMETER_RANGES = {
                'beta_0': (0.5, 2.0),
                'sigma': (0.5, 2.0),
                'gamma': (0.5, 2.0),
                'alpha_T': (0.5, 2.0),
                'kappa': (0.0, 1.0),
                'k_0': (0.5, 2.0),
                'alpha_net': (0.5, 2.0),
                'beta_ep': (0.0, 0.2)
            }
        
        if self.PARAMETER_UNCERTAINTIES is None:
            self.PARAMETER_UNCERTAINTIES = {
                'beta_0': 0.2,
                'sigma': 0.1,
                'gamma': 0.1,
                'alpha_T': 0.3,
                'kappa': 0.1,
                'k_0': 0.5,
                'alpha_net': 0.2,
                'beta_ep': 0.05
            }
    
    @classmethod
    def from_environment(cls):
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv('N_SAMPLES'):
            config.DEFAULT_SAMPLES = int(os.getenv('N_SAMPLES'))
        
        if os.getenv('MC_SAMPLES'):
            config.MC_DEFAULT_SAMPLES = int(os.getenv('MC_SAMPLES'))
        
        if os.getenv('TIMEOUT_HOURS'):
            config.TIMEOUT_HOURS = float(os.getenv('TIMEOUT_HOURS'))
        
        if os.getenv('MAX_TIME'):
            config.MAX_TIME = int(os.getenv('MAX_TIME'))
        
        if os.getenv('SIMULATION_DAYS'):
            config.SIMULATION_DAYS = int(os.getenv('SIMULATION_DAYS'))
        
        return config
    
    def get_optimal_sample_size(self, requested_size: int, quick_mode: bool = False) -> int:
        """Get optimal sample size (power of 2) based on configuration"""
        if quick_mode:
            base_size = self.QUICK_MODE_SAMPLES
        else:
            base_size = requested_size or self.DEFAULT_SAMPLES
        
        # Ensure within bounds
        if base_size < self.MIN_SAMPLES:
            base_size = self.MIN_SAMPLES
        elif base_size > self.MAX_SAMPLES:
            base_size = self.MAX_SAMPLES
        
        # Round to nearest power of 2
        power = int(np.log2(base_size))
        lower = 2**power
        upper = 2**(power + 1)
        
        return lower if base_size - lower < upper - base_size else upper
    
    def get_mc_sample_size(self, requested_size: int, quick_mode: bool = False) -> int:
        """Get optimal Monte Carlo sample size"""
        if quick_mode:
            base_size = self.MC_QUICK_MODE_SAMPLES
        else:
            base_size = requested_size or self.MC_DEFAULT_SAMPLES
        
        return self.get_optimal_sample_size(base_size, quick_mode)

# Import numpy for the log2 function
import numpy as np 