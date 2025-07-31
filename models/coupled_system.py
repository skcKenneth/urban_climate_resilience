"""
Coupled epidemic-network dynamics system
"""
import numpy as np
from scipy.integrate import solve_ivp
import networkx as nx
from models.epidemiological_model import ClimateEpidemiologicalModel
from models.network_model import DynamicNetworkModel
from utils.parameters import ModelParameters

class CoupledSystemModel:
    """Integration of epidemic and network dynamics with climate forcing"""
    
    def __init__(self, params=None, n_nodes=1000):
        self.params = params if params else ModelParameters()
        self.epi_model = ClimateEpidemiologicalModel(params)
        self.net_model = DynamicNetworkModel(params, n_nodes)
        self.n_nodes = n_nodes
        
    def target_average_degree(self, T):
        """Target average degree under climate conditions"""
        return self.params.k_0 * np.exp(
            -(T - self.params.T_opt)**2 / (2 * self.params.sigma_k**2)
        )
    
    def target_clustering(self, T, I_frac):
        """Target clustering coefficient"""
        base_clustering = 0.3
        temp_effect = np.exp(-(T - self.params.T_opt)**2 / 50)
        epidemic_effect = np.exp(-5 * I_frac)
        return base_clustering * temp_effect * epidemic_effect
    
    def coupled_derivatives(self, t, y, T_func, H_func, controls=None):
        """Derivatives for coupled system"""
        # State variables
        S, E, I, R, k_avg, C = y
        N = self.params.N
        
        # Climate conditions
        T = T_func(t)
        H = H_func(t)
        
        # Current infection fraction
        I_frac = I / N
        
        # Control inputs
        if controls is not None:
            u1, u2, u3 = controls(t)
        else:
            u1 = u2 = u3 = 0
        
        # Modified transmission rate with network and control effects
        beta_base = self.epi_model.beta_climate(T, H) * self.epi_model.social_behavior_factor(T)
        beta = beta_base * (k_avg / self.params.k_0) * (1 - 0.3 * u1)  # Medical intervention
        
        # Modified death rate
        delta = self.epi_model.death_rate_climate(T) * (1 - 0.5 * u1)
        
        # SEIR equations
        dSdt = self.params.mu * N - beta * S * I / N - self.params.mu * S
        dEdt = beta * S * I / N - (self.params.sigma + self.params.mu) * E
        dIdt = self.params.sigma * E - (self.params.gamma + self.params.mu + delta) * I
        dRdt = self.params.gamma * I - self.params.mu * R
        
        # Network dynamics
        k_target = self.target_average_degree(T - 2 * u3)  # Climate mitigation effect
        dk_avg_dt = (self.params.alpha_net * (k_target - k_avg) - 
                     self.params.beta_ep * I_frac + 0.2 * u2)  # Social support effect
        
        # Clustering dynamics
        C_target = self.target_clustering(T, I_frac)
        dCdt = self.params.gamma_cluster * (C_target - C) * (1 + 0.3 * u2)
        
        return [dSdt, dEdt, dIdt, dRdt, dk_avg_dt, dCdt]
    
    def solve_coupled_system(self, t_span, y0, T_func, H_func, controls=None):
        """Solve the coupled system"""
        def rhs(t, y):
            return self.coupled_derivatives(t, y, T_func, H_func, controls)
        
        sol = solve_ivp(rhs, t_span, y0,
                       t_eval=np.arange(t_span[0], t_span[1], self.params.dt),
                       method='RK45', rtol=1e-6)
        
        return sol.t, sol.y
    
    def simulate_with_stochasticity(self, t_span, y0, T_func, H_func, 
                                   controls=None, noise_scale=0.01):
        """Simulate with stochastic perturbations"""
        t_eval = np.arange(t_span[0], t_span[1], self.params.dt)
        n_steps = len(t_eval)
        n_vars = len(y0)
        
        # Storage arrays
        y_trajectory = np.zeros((n_vars, n_steps))
        y_trajectory[:, 0] = y0
        
        # Integration with noise
        for i in range(1, n_steps):
            t = t_eval[i-1]
            dt = self.params.dt
            y_curr = y_trajectory[:, i-1]
            
            # Deterministic derivatives
            dydt = self.coupled_derivatives(t, y_curr, T_func, H_func, controls)
            
            # Add stochastic perturbations
            noise = np.random.normal(0, noise_scale, n_vars)
            noise[0:4] *= np.sqrt(y_curr[0:4] + 1)  # Demographic stochasticity
            
            # Update
            y_new = y_curr + np.array(dydt) * dt + noise * np.sqrt(dt)
            
            # Ensure non-negative populations
            y_new[0:4] = np.maximum(y_new[0:4], 0)
            y_new[4] = max(y_new[4], 1.0)  # Minimum average degree
            y_new[5] = np.clip(y_new[5], 0, 1)  # Clustering coefficient bounds
            
            y_trajectory[:, i] = y_new
        
        return t_eval, y_trajectory
    
    def calculate_system_resilience(self, t, y, T_func):
        """Calculate system resilience metrics"""
        S, E, I, R, k_avg, C = y
        N = self.params.N
        
        # Health resilience
        health_impact = (E + I) / N
        
        # Network resilience
        k_baseline = self.params.k_0
        network_integrity = min(k_avg / k_baseline, 1.0)
        
        # Social cohesion
        social_cohesion = C
        
        # Overall resilience (harmonic mean)
        resilience_components = [1 - health_impact, network_integrity, social_cohesion]
        overall_resilience = len(resilience_components) / sum(1/max(r, 0.01) for r in resilience_components)
        
        return {
            'health_resilience': 1 - health_impact,
            'network_resilience': network_integrity,
            'social_resilience': social_cohesion,
            'overall_resilience': overall_resilience
        }
