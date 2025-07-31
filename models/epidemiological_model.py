"""
Climate-forced epidemiological model implementation
"""
import numpy as np
from scipy.integrate import solve_ivp
from utils.parameters import ModelParameters

class ClimateEpidemiologicalModel:
    """SEIR model with climate forcing and social behavior modifications"""
    
    def __init__(self, params=None):
        self.params = params if params else ModelParameters()
        
    def beta_climate(self, T, H):
        """Temperature and humidity dependent transmission rate"""
        temp_effect = 1 + self.params.alpha_T * (T - self.params.T_0)**2
        humidity_effect = 1 + self.params.alpha_H * (H - self.params.H_0)
        return self.params.beta_0 * temp_effect * humidity_effect
    
    def social_behavior_factor(self, T):
        """Social distancing behavior as function of temperature"""
        if T > self.params.T_critical:
            return 1 - self.params.kappa * np.tanh(
                self.params.theta * (T - self.params.T_critical)
            )
        return 1.0
    
    def death_rate_climate(self, T):
        """Climate-dependent death rate"""
        return self.params.delta_0 * (1 + 0.1 * (T - self.params.T_0)**2 / 100)
    
    def seir_derivatives(self, t, y, T_func, H_func, k_avg=None):
        """SEIR system derivatives with climate forcing"""
        S, E, I, R = y
        N = self.params.N
        
        # Get climate conditions
        T = T_func(t)
        H = H_func(t)
        
        # Calculate rates
        beta = self.beta_climate(T, H) * self.social_behavior_factor(T)
        if k_avg is not None:
            # Network effect on transmission
            beta *= k_avg(t) / self.params.k_0
        
        delta = self.death_rate_climate(T)
        
        # SEIR equations
        dSdt = self.params.mu * N - beta * S * I / N - self.params.mu * S
        dEdt = beta * S * I / N - (self.params.sigma + self.params.mu) * E
        dIdt = self.params.sigma * E - (self.params.gamma + self.params.mu + delta) * I
        dRdt = self.params.gamma * I - self.params.mu * R
        
        return [dSdt, dEdt, dIdt, dRdt]
    
    def solve(self, t_span, y0, T_func, H_func, k_avg_func=None):
        """Solve the SEIR system"""
        def rhs(t, y):
            return self.seir_derivatives(t, y, T_func, H_func, k_avg_func)
        
        sol = solve_ivp(rhs, t_span, y0, 
                       t_eval=np.arange(t_span[0], t_span[1], self.params.dt),
                       method='RK45', rtol=1e-6)
        
        return sol.t, sol.y
    
    def basic_reproduction_number(self, T, H, k_avg=1.0):
        """Calculate R0 as function of climate and network conditions"""
        beta = self.beta_climate(T, H) * self.social_behavior_factor(T)
        delta = self.death_rate_climate(T)
        
        R0 = (beta * self.params.sigma * k_avg) / (
            (self.params.sigma + self.params.mu) * 
            (self.params.gamma + self.params.mu + delta)
        )
        return R0
    
    def equilibrium_analysis(self, T_range, H=0.6):
        """Analyze equilibria across temperature range"""
        R0_values = []
        for T in T_range:
            R0_values.append(self.basic_reproduction_number(T, H))
        return np.array(R0_values)
