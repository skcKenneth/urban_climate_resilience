"""
Simple epidemic model for climate-epidemic analysis
"""
import numpy as np
from scipy.integrate import odeint

class EpidemicModel:
    """SEIR epidemic model with climate coupling"""
    
    def __init__(self):
        # Basic SEIR parameters
        self.beta_0 = 0.5    # Base transmission rate
        self.sigma = 1/5.1   # Incubation rate (5.1 days)
        self.gamma = 1/7     # Recovery rate (7 days)
        self.mu = 0.01       # Death rate
        
        # Climate coupling
        self.alpha_T = 0.02  # Temperature effect on transmission
        self.T_ref = 20      # Reference temperature
        
        # Initial conditions
        self.N = 1.0         # Total population (normalized)
        self.I0 = 0.001      # Initial infected
        self.E0 = 0.001      # Initial exposed
        self.R0 = 0.0        # Initial recovered
        self.S0 = self.N - self.I0 - self.E0 - self.R0
    
    def transmission_rate(self, T):
        """Temperature-dependent transmission rate"""
        return self.beta_0 * (1 + self.alpha_T * (T - self.T_ref))
    
    def seir_dynamics(self, y, t, T_func):
        """SEIR differential equations"""
        S, E, I, R = y
        T = T_func(t) if callable(T_func) else T_func
        
        beta = self.transmission_rate(T)
        
        dS = -beta * S * I
        dE = beta * S * I - self.sigma * E
        dI = self.sigma * E - self.gamma * I - self.mu * I
        dR = self.gamma * I
        
        return [dS, dE, dI, dR]
    
    def simulate(self, t, T_func=None):
        """Run SEIR simulation"""
        if T_func is None:
            T_func = lambda t: self.T_ref
        
        y0 = [self.S0, self.E0, self.I0, self.R0]
        solution = odeint(self.seir_dynamics, y0, t, args=(T_func,))
        
        return solution