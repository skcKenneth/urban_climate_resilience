"""
Optimal control for resource allocation in climate-epidemic system
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from models.coupled_system import CoupledSystemModel
from utils.parameters import ModelParameters

class OptimalControlModel:
    """Optimal resource allocation for coupled system"""
    
    def __init__(self, params=None):
        self.params = params if params else ModelParameters()
        self.coupled_model = CoupledSystemModel(params)
        
    def objective_function(self, control_params, t_span, y0, T_func, H_func, budget_func):
        """Objective function for optimal control"""
        
        # Reshape control parameters into time-dependent controls
        n_time_points = int((t_span[1] - t_span[0]) / self.params.dt)
        n_controls = 3
        
        control_matrix = control_params.reshape((n_controls, -1))
        
        def controls(t):
            # Linear interpolation of control values
            t_idx = min(int((t - t_span[0]) / self.params.dt), control_matrix.shape[1] - 1)
            return [max(0, min(control_matrix[i, t_idx], self.params.u_max[i])) 
                   for i in range(n_controls)]
        
        # Simulate system with controls
        try:
            t, y = self.coupled_model.solve_coupled_system(t_span, y0, T_func, H_func, controls)
        except:
            return 1e6  # Large penalty for failed integration
        
        # Calculate objective components
        S, E, I, R, k_avg, C = y
        N = self.params.N
        
        # Health cost (infections and deaths)
        health_cost = np.trapz(I, t)
        
        # Social disruption cost
        k_baseline = self.params.k_0
        social_cost = np.trapz(np.maximum(0, k_baseline - k_avg)**2, t)
        
        # Control costs
        control_cost = 0
        for i in range(len(t)):
            u_vals = controls(t[i])
            control_cost += sum(self.params.costs[j] * u_vals[j]**2 
                              for j in range(n_controls)) * self.params.dt
        
        # Budget constraint penalty
        budget_penalty = 0
        for i in range(len(t)):
            u_vals = controls(t[i])
            total_cost = sum(self.params.costs[j] * u_vals[j] for j in range(n_controls))
            if total_cost > budget_func(t[i]):
                budget_penalty += (total_cost - budget_func(t[i]))**2 * 1000
        
        # Total objective
        objective = (self.params.weights[0] * health_cost + 
                    self.params.weights[1] * social_cost + 
                    self.params.weights[2] * control_cost +
                    budget_penalty)
        
        return objective
    
    def solve_optimal_control(self, t_span, y0, T_func, H_func, budget_func, 
                            optimization_method='differential_evolution'):
        """Solve optimal control problem"""
        
        n_time_points = max(10, int((t_span[1] - t_span[0]) / (self.params.dt * 10)))
        n_controls = 3
        n_vars = n_controls * n_time_points
        
        # Bounds for control variables
        bounds = [(0, self.params.u_max[i % n_controls]) for i in range(n_vars)]
        
        if optimization_method == 'differential_evolution':
            result = differential_evolution(
                self.objective_function,
                bounds,
                args=(t_span, y0, T_func, H_func, budget_func),
                seed=42,
                maxiter=50,
                popsize=10
            )
        else:
            # Initial guess
            x0 = np.random.uniform(0, 0.5, n_vars)
            
            result = minimize(
                self.objective_function,
                x0,
                args=(t_span, y0, T_func, H_func, budget_func),
                bounds=bounds,
                method='L-BFGS-B'
            )
        
        # Convert result to control functions
        optimal_controls = result.x.reshape((n_controls, -1))
        
        def optimal_control_func(t):
            t_idx = min(int((t - t_span[0]) / self.params.dt / 10), 
                       optimal_controls.shape[1] - 1)
            return [max(0, min(optimal_controls[i, t_idx], self.params.u_max[i])) 
                   for i in range(n_controls)]
        
        return optimal_control_func, result.fun, result
    
    def compare_strategies(self, t_span, y0, T_func, H_func, budget_func):
        """Compare different control strategies"""
        
        strategies = {
            'no_control': lambda t: [0, 0, 0],
            'medical_only': lambda t: [min(0.8, budget_func(t)/self.params.costs[0]), 0, 0],
            'social_only': lambda t: [0, min(0.8, budget_func(t)/self.params.costs[1]), 0],
            'climate_only': lambda t: [0, 0, min(0.8, budget_func(t)/self.params.costs[2])],
        }
        
        # Add optimal strategy
        optimal_control, _, _ = self.solve_optimal_control(t_span, y0, T_func, H_func, budget_func)
        strategies['optimal'] = optimal_control
        
        results = {}
        
        for name, control_func in strategies.items():
            try:
                t, y = self.coupled_model.solve_coupled_system(
                    t_span, y0, T_func, H_func, control_func
                )
                
                S, E, I, R, k_avg, C = y
                
                results[name] = {
                    't': t,
                    'y': y,
                    'total_infections': np.trapz(I, t),
                    'peak_infections': np.max(I),
                    'final_network_degree': k_avg[-1],
                    'avg_clustering': np.mean(C),
                    'resilience_scores': [
                        self.coupled_model.calculate_system_resilience(
                            t[i], y[:, i], T_func
                        )['overall_resilience'] for i in range(len(t))
                    ]
                }
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results
