"""
Climate model for temperature scenarios
"""
import numpy as np

class ClimateModel:
    """Simple climate model for different scenarios"""
    
    def __init__(self):
        self.T_base = 20.0      # Base temperature (°C)
        self.T_amp = 10.0       # Seasonal amplitude
        self.scenario = 'baseline'
        
    def temperature(self, t):
        """Generate temperature time series based on scenario"""
        # Basic seasonal variation
        T = self.T_base + self.T_amp * np.sin(2 * np.pi * t / 365)
        
        if self.scenario == 'heatwave':
            # Add heatwave effect (5°C increase in summer)
            summer_mask = np.sin(2 * np.pi * t / 365) > 0.5
            T = T + 5 * summer_mask
            
        elif self.scenario == 'extreme':
            # Add extreme weather variability
            T = T + 5 * np.sin(4 * np.pi * t / 365) + np.random.normal(0, 2, size=np.shape(t))
            
        return T