"""
Stability and bifurcation analysis for the coupled system
"""
import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
from models.coupled_system import CoupledSystemModel
from utils.parameters import ModelParameters

class StabilityAnalysis:
    """Stability and bifurcation analysis tools"""
    
    def __init__(self, params=None):
        self.params = params if params else ModelParameters()
        self.coupled_model = CoupledSystemModel(params)
        
    def find_equilibria(self, T, H, control_vals=[0, 0, 0]):
        """Find equilibrium points for given climate conditions"""
        
        def equilibrium_equations(y):
            S, E, I, R, k_avg, C = y
            
            # Constant climate functions
            T_func = lambda t: T
            H_func = lambda t: H
            controls = lambda t: control_vals
            
            # Get derivatives at equilibrium (should be zero)
            dydt = self.coupled_model.coupled_derivatives(0, y, T_func, H_func, controls)
            return dydt
        
        # Multiple initial guesses for finding different equilibria
        initial_guesses = [
            [self.params.N*0.9, 0, 0, 0, self.params.k_0, 0.3],  # Disease-free
            [self.params.N*0.6, self.params.N*0.1, self.params.N*0.2, self.params.N*0.1, 
             self.params.k_0*0.8, 0.2],  # Endemic
        ]
        
        equilibria = []
        for guess in initial_guesses:
            try:
                eq = fsolve(equilibrium_equations, guess, xtol=1e-8)
                # Verify it's actually an equilibrium
                residual = equilibrium_equations(eq)
                if np.linalg.norm(residual) < 1e-6:
                    # Check if we already found this equilibrium
                    is_new = True
                    for existing_eq in equilibria:
                        if np.linalg.norm(np.array(eq) - np.array(existing_eq)) < 1e-4:
                            is_new = False
                            break
                    if is_new:
                        equilibria.append(eq.tolist())
            except:
                continue
                
        return equilibria
    
    def jacobian_matrix(self, y, T, H, control_vals=[0, 0, 0]):
        """Compute Jacobian matrix at given state"""
        
        eps = 1e-8
        n = len(y)
        J = np.zeros((n, n))
        
        T_func = lambda t: T
        H_func = lambda t: H
        controls = lambda t: control_vals
        
        # Base derivatives
        f0 = self.coupled_model.coupled_derivatives(0, y, T_func, H_func, controls)
        
        # Numerical differentiation
        for j in range(n):
            y_plus = y.copy()
            y_plus[j] += eps
            f_plus = self.coupled_model.coupled_derivatives(0, y_plus, T_func, H_func, controls)
            J[:, j] = (np.array(f_plus) - np.array(f0)) / eps
            
        return J
    
    def stability_analysis(self, equilibrium, T, H, control_vals=[0, 0, 0]):
        """Analyze stability of equilibrium point"""
        
        J = self.jacobian_matrix(equilibrium, T, H, control_vals)
        eigenvalues = eigvals(J)
        
        # Classify stability
        max_real_part = np.max(np.real(eigenvalues))
        
        if max_real_part < -1e-8:
            stability = 'stable'
        elif max_real_part > 1e-8:
            stability = 'unstable'
        else:
            stability = 'marginal'
        
        return {
            'eigenvalues': eigenvalues,
            'max_real_part': max_real_part,
            'stability': stability,
            'jacobian': J
        }
    
    def bifurcation_analysis_temperature(self, T_range, H=0.6, control_vals=[0, 0, 0]):
        """Perform bifurcation analysis over temperature range"""
        
        results = {
            'temperatures': [],
            'equilibria': [],
            'stability': [],
            'eigenvalues': [],
            'R0_values': []
        }
        
        for T in T_range:
            try:
                # Find equilibria
                eq_points = self.find_equilibria(T, H, control_vals)
                
                for eq in eq_points:
                    # Stability analysis
                    stability_info = self.stability_analysis(eq, T, H, control_vals)
                    
                    # Calculate R0
                    S, E, I, R = eq[:4]
                    k_avg = eq[4]
                    R0 = self.coupled_model.epi_model.basic_reproduction_number(T, H, k_avg)
                    
                    results['temperatures'].append(T)
                    results['equilibria'].append(eq)
                    results['stability'].append(stability_info['stability'])
                    results['eigenvalues'].append(stability_info['eigenvalues'])
                    results['R0_values'].append(R0)
                    
            except Exception as e:
                print(f"Error at T={T}: {e}")
                continue
        
        return results
    
    def critical_temperature_analysis(self, T_range, H=0.6):
        """Identify critical temperatures where bifurcations occur"""
        
        bifurcation_results = self.bifurcation_analysis_temperature(T_range, H)
        
        # Look for stability changes
        critical_points = []
        
        temperatures = np.array(bifurcation_results['temperatures'])
        R0_values = np.array(bifurcation_results['R0_values'])
        
        # Find where R0 crosses 1
        for i in range(len(R0_values)-1):
            if (R0_values[i] - 1) * (R0_values[i+1] - 1) < 0:
                T_critical = temperatures[i] + (temperatures[i+1] - temperatures[i]) * \
                            (1 - R0_values[i]) / (R0_values[i+1] - R0_values[i])
                critical_points.append({
                    'temperature': T_critical,
                    'type': 'transcritical',
                    'R0': 1.0
                })
        
        return critical_points
    
    def sensitivity_to_parameters(self, parameter_variations, T=30.0, H=0.6):
        """Analyze sensitivity of system behavior to parameter variations"""
        
        base_params = ModelParameters()
        results = {}
        
        # Base case
        base_equilibria = self.find_equilibria(T, H)
        if base_equilibria:
            base_eq = base_equilibria[0]  # Take first equilibrium
            base_stability = self.stability_analysis(base_eq, T, H)
            results['base'] = {
                'equilibrium': base_eq,
                'stability': base_stability['stability'],
                'max_eigenvalue': base_stability['max_real_part']
            }
        
        # Parameter variations
        for param_name, variation_range in parameter_variations.items():
            results[param_name] = []
            
            for variation in variation_range:
                # Create modified parameters
                modified_params = ModelParameters()
                setattr(modified_params, param_name, 
                       getattr(base_params, param_name) * variation)
                
                # Create new model with modified parameters
                modified_model = CoupledSystemModel(modified_params)
                
                # Temporarily replace the model in stability analyzer
                original_model = self.coupled_model
                self.coupled_model = modified_model
                
                try:
                    equilibria = self.find_equilibria(T, H)
                    if equilibria:
                        eq = equilibria[0]
                        stability_info = self.stability_analysis(eq, T, H)
                        results[param_name].append({
                            'variation': variation,
                            'equilibrium': eq,
                            'stability': stability_info['stability'],
                            'max_eigenvalue': stability_info['max_real_part']
                        })
                except:
                    results[param_name].append({
                        'variation': variation,
                        'error': 'Failed to find equilibrium'
                    })
                
                # Restore original model
                self.coupled_model = original_model
        
        return results
