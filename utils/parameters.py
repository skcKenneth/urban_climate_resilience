"""
Parameter definitions for the Urban Climate-Social Network Resilience System
"""
import numpy as np

class ModelParameters:
    """Central parameter storage for all models"""
    
    def __init__(self):
        # Population and demographic parameters
        self.N = 100000  # Total population
        self.mu = 1/(70*365)  # Natural death rate (per day)
        
        # Epidemiological parameters
        self.beta_0 = 0.3  # Base transmission rate
        self.sigma = 1/5.1  # Incubation rate (1/incubation period)
        self.gamma = 1/10  # Recovery rate (1/infectious period)
        self.delta_0 = 0.01  # Base disease-induced death rate
        
        # Climate forcing parameters
        self.alpha_T = 0.02  # Temperature effect coefficient
        self.alpha_H = 0.01  # Humidity effect coefficient
        self.T_0 = 25.0  # Reference temperature (°C)
        self.H_0 = 0.6  # Reference humidity
        self.T_critical = 35.0  # Critical temperature threshold
        self.T_opt = 22.0  # Optimal temperature for social interaction
        self.sigma_T = 5.0  # Temperature sensitivity parameter
        
        # Social behavior parameters
        self.kappa = 0.4  # Social distancing coefficient
        self.theta = 0.5  # Social response steepness
        
        # Network parameters
        self.k_0 = 8.0  # Base average degree
        self.sigma_k = 3.0  # Network temperature sensitivity
        self.lambda_dist = 0.1  # Distance decay parameter
        self.xi = 0.02  # Base edge dissolution rate
        self.h_0 = 0.01  # Base stress dissolution rate
        self.T_threshold = 30.0  # Network stress threshold
        self.delta_T = 2.0  # Temperature transition width
        
        # Coupling parameters
        self.alpha_net = 0.1  # Network adaptation rate
        self.beta_ep = 0.05  # Epidemic impact on network
        self.gamma_cluster = 0.08  # Clustering adaptation rate
        
        # Control parameters
        self.u_max = [1.0, 1.0, 1.0]  # Maximum control intensities
        self.costs = [100, 50, 200]  # Control costs
        self.weights = [1.0, 0.1, 0.01]  # Objective weights
        
        # Simulation parameters
        self.dt = 0.1  # Time step (days)
        self.T_sim = 365  # Simulation time (days)
        
    def get_climate_scenario(self, scenario='baseline'):
        """Generate temperature and humidity time series"""
        t = np.arange(0, self.T_sim, self.dt)
        
        if scenario == 'baseline':
            T = 25 + 10*np.sin(2*np.pi*t/365) + 2*np.random.normal(0, 1, len(t))
            H = 0.6 + 0.2*np.sin(2*np.pi*t/365 + np.pi/4)
        elif scenario == 'heatwave':
            T = 25 + 10*np.sin(2*np.pi*t/365) + 2*np.random.normal(0, 1, len(t))
            # Add heatwave events
            heatwave_times = [100, 200, 300]
            for hw_time in heatwave_times:
                mask = (t >= hw_time) & (t <= hw_time + 14)
                T[mask] += 8*np.exp(-(t[mask] - hw_time - 7)**2/20)
            H = 0.6 + 0.2*np.sin(2*np.pi*t/365 + np.pi/4)
        
        return t, T, H
