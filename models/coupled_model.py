"""
Coupled climate-epidemic-network model
"""
import numpy as np
from scipy.integrate import odeint

class CoupledClimateEpidemicNetwork:
    """Coupled model integrating climate, epidemic, and network dynamics"""
    
    def __init__(self, epidemic_model, climate_model, network_model=None):
        self.epidemic = epidemic_model
        self.climate = climate_model
        self.network = network_model
        
    def coupled_dynamics(self, y, t):
        """Coupled differential equations"""
        # Split state vector
        n_epidemic = 4  # S, E, I, R
        epidemic_state = y[:n_epidemic]
        
        # Get temperature
        T = self.climate.temperature(t)
        
        # Update epidemic dynamics with temperature
        epidemic_derivs = self.epidemic.seir_dynamics(epidemic_state, t, lambda t: T)
        
        # If network model exists, add network dynamics
        if self.network is not None and len(y) > n_epidemic:
            network_state = y[n_epidemic:]
            # Simple network degradation based on infection
            I = epidemic_state[2]
            network_derivs = [-0.1 * I * network_state[0]]  # k_avg decreases with infection
            return epidemic_derivs + network_derivs
        else:
            return epidemic_derivs
    
    def simulate(self, t):
        """Run coupled simulation"""
        # Initial conditions
        epidemic_y0 = [self.epidemic.S0, self.epidemic.E0, 
                       self.epidemic.I0, self.epidemic.R0]
        
        if self.network is not None:
            network_y0 = [10.0]  # Initial average degree
            y0 = epidemic_y0 + network_y0
        else:
            y0 = epidemic_y0
        
        # Solve ODEs
        solution = odeint(self.coupled_dynamics, y0, t)
        
        # Add temperature and dummy network data if needed
        T = np.array([self.climate.temperature(t)]).T
        
        if solution.shape[1] == 4:  # Only epidemic data
            # Add dummy network data
            k_avg = 10 * np.ones((len(t), 1))
            solution = np.hstack([solution, T, k_avg])
        else:
            solution = np.hstack([solution, T])
        
        return solution