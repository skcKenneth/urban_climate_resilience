"""
Optimal control for resource allocation in climate-epidemic system
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from models.coupled_system import CoupledSystemModel
from utils.parameters import ModelParameters
import multiprocessing as mp
from functools import partial
import time

class OptimalControlModel:
    """Optimal resource allocation for coupled system"""
    
    def __init__(self, params=None):
        self.params = params if params else ModelParameters()
        self.coupled_model = CoupledSystemModel(params)
        self.max_runtime = 300  # 5 minutes max for optimal control
        self.start_time = None
        
    def check_timeout(self):
        """Check if optimization has exceeded time limit"""
        if self.start_time and (time.time() - self.start_time) > self.max_runtime:
            raise TimeoutError("Optimal control optimization exceeded time limit")
        
    def adaptive_discretization(self, t_span, complexity='low'):
        """Adaptive time discretization based on problem complexity"""
        duration = t_span[1] - t_span[0]
        
        if complexity == 'low':
            # Coarse discretization for quick analysis
            n_points = min(5, max(3, int(duration / 30)))
        elif complexity == 'medium':
            # Medium discretization
            n_points = min(10, max(5, int(duration / 15)))
        else:
            # Fine discretization for detailed analysis
            n_points = min(20, max(10, int(duration / 7)))
            
        return n_points
        
    def objective_function(self, control_params, t_span, y0, T_func, H_func, budget_func):
        """Objective function for optimal control with early stopping"""
        
        # Check timeout
        try:
            self.check_timeout()
        except TimeoutError:
            return 1e6
        
        # Reshape control parameters into time-dependent controls
        n_controls = 3
        control_matrix = control_params.reshape((n_controls, -1))
        n_time_points = control_matrix.shape[1]
        
        # Create interpolation functions for smooth controls
        t_control = np.linspace(t_span[0], t_span[1], n_time_points)
        control_interps = []
        for i in range(n_controls):
            control_interps.append(interp1d(t_control, control_matrix[i, :], 
                                          kind='linear', bounds_error=False, 
                                          fill_value=(control_matrix[i, 0], control_matrix[i, -1])))
        
        def controls(t):
            return [max(0, min(control_interps[i](t), self.params.u_max[i])) 
                   for i in range(n_controls)]
        
        # Simulate system with controls - use coarser time grid for faster evaluation
        try:
            # Use adaptive time stepping
            t_eval = np.linspace(t_span[0], t_span[1], min(50, int((t_span[1] - t_span[0]) / 2)))
            t, y = self.coupled_model.solve_coupled_system(t_span, y0, T_func, H_func, controls, t_eval=t_eval)
        except Exception as e:
            return 1e6  # Large penalty for failed integration
        
        # Calculate objective components
        S, E, I, R, k_avg, C = y
        N = self.params.N
        
        # Health cost (infections and deaths)
        health_cost = np.trapz(I, t) / N
        
        # Social disruption cost
        k_baseline = self.params.k_0
        social_cost = np.trapz(np.maximum(0, (k_baseline - k_avg) / k_baseline)**2, t) / len(t)
        
        # Control costs - use Simpson's rule for better accuracy with fewer points
        control_cost = 0
        t_ctrl_eval = np.linspace(t_span[0], t_span[1], min(20, len(t)))
        for tc in t_ctrl_eval:
            u_vals = controls(tc)
            control_cost += sum(self.params.costs[j] * u_vals[j]**2 
                              for j in range(n_controls))
        control_cost *= (t_span[1] - t_span[0]) / len(t_ctrl_eval)
        
        # Budget constraint penalty - check at key points only
        budget_penalty = 0
        for tc in t_ctrl_eval[::2]:  # Check every other point
            u_vals = controls(tc)
            total_cost = sum(self.params.costs[j] * u_vals[j] for j in range(n_controls))
            if total_cost > budget_func(tc):
                budget_penalty += (total_cost - budget_func(tc))**2 * 100
        
        # Total objective with normalized weights
        objective = (self.params.weights[0] * health_cost + 
                    self.params.weights[1] * social_cost + 
                    self.params.weights[2] * control_cost / 1000 +
                    budget_penalty)
        
        return objective
    
    def solve_optimal_control(self, t_span, y0, T_func, H_func, budget_func, 
                            optimization_method='differential_evolution',
                            complexity='medium'):
        """Solve optimal control problem with adaptive complexity"""
        
        self.start_time = time.time()
        
        # Adaptive discretization based on complexity
        n_time_points = self.adaptive_discretization(t_span, complexity)
        n_controls = 3
        n_vars = n_controls * n_time_points
        
        # Bounds for control variables
        bounds = [(0, self.params.u_max[i % n_controls]) for i in range(n_vars)]
        
        # Optimization parameters based on complexity
        if complexity == 'low':
            maxiter, popsize = 20, 5
        elif complexity == 'medium':
            maxiter, popsize = 40, 10
        else:
            maxiter, popsize = 80, 15
        
        if optimization_method == 'differential_evolution':
            # Use parallel evaluation if available
            workers = min(mp.cpu_count() - 1, 4) if complexity != 'low' else 1
            
            result = differential_evolution(
                self.objective_function,
                bounds,
                args=(t_span, y0, T_func, H_func, budget_func),
                seed=42,
                maxiter=maxiter,
                popsize=popsize,
                workers=workers,
                updating='deferred',
                disp=False,
                polish=False  # Skip polishing for faster results
            )
        else:
            # Initial guess - use simple heuristic
            x0 = np.zeros(n_vars)
            # Start with moderate medical control
            x0[::n_controls] = 0.3
            
            result = minimize(
                self.objective_function,
                x0,
                args=(t_span, y0, T_func, H_func, budget_func),
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-6}
            )
        
        # Convert result to control functions
        optimal_controls = result.x.reshape((n_controls, -1))
        t_control = np.linspace(t_span[0], t_span[1], n_time_points)
        
        # Create smooth interpolation functions
        control_interps = []
        for i in range(n_controls):
            control_interps.append(interp1d(t_control, optimal_controls[i, :], 
                                          kind='linear', bounds_error=False,
                                          fill_value=(optimal_controls[i, 0], optimal_controls[i, -1])))
        
        def optimal_control_func(t):
            return [max(0, min(control_interps[i](t), self.params.u_max[i])) 
                   for i in range(n_controls)]
        
        return optimal_control_func, result.fun, result
    
    def compare_strategies(self, t_span, y0, T_func, H_func, budget_func, quick_mode=False):
        """Compare different control strategies with optional quick mode"""
        
        strategies = {
            'no_control': lambda t: [0, 0, 0],
            'medical_only': lambda t: [min(0.8, budget_func(t)/self.params.costs[0]), 0, 0],
            'social_only': lambda t: [0, min(0.8, budget_func(t)/self.params.costs[1]), 0],
            'climate_only': lambda t: [0, 0, min(0.8, budget_func(t)/self.params.costs[2])],
        }
        
        # Add optimal strategy with complexity based on mode
        complexity = 'low' if quick_mode else 'medium'
        try:
            optimal_control, _, _ = self.solve_optimal_control(
                t_span, y0, T_func, H_func, budget_func, complexity=complexity
            )
            strategies['optimal'] = optimal_control
        except Exception as e:
            print(f"Warning: Optimal control failed with {e}, using balanced strategy")
            # Fallback balanced strategy
            strategies['optimal'] = lambda t: [
                min(0.3, budget_func(t)/(3*self.params.costs[0])),
                min(0.3, budget_func(t)/(3*self.params.costs[1])),
                min(0.3, budget_func(t)/(3*self.params.costs[2]))
            ]
        
        results = {}
        
        # Use coarser time grid for comparison
        t_eval = np.linspace(t_span[0], t_span[1], 100 if quick_mode else 200)
        
        for name, control_func in strategies.items():
            try:
                t, y = self.coupled_model.solve_coupled_system(
                    t_span, y0, T_func, H_func, control_func, t_eval=t_eval
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
                        )['overall_resilience'] for i in range(0, len(t), max(1, len(t)//50))
                    ],
                    'control_values': [control_func(ti) for ti in t[::max(1, len(t)//50)]]
                }
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results
