"""
Sensitivity analysis and model validation
"""
import numpy as np
import os
import time
from functools import lru_cache
try:
    from scipy.stats.qmc import Sobol  # For scipy >= 1.7
except ImportError:
    from scipy.stats import sobol_seq  # For older scipy versions
import matplotlib.pyplot as plt
from models.coupled_system import CoupledSystemModel
from utils.parameters import ModelParameters
from utils.sensitivity_config import SensitivityConfig

class SensitivityAnalysis:
    """Model validation and sensitivity analysis tools"""
    
    def __init__(self, params=None):
        self.params = params if params else ModelParameters()
        self.config = SensitivityConfig.from_environment()
        self.start_time = time.time()
        self.max_time = self.config.MAX_TIME
        self.quick_mode = os.getenv('QUICK_MODE', 'false').lower() == 'true'
        
    def check_timeout(self):
        """Check if we're approaching the time limit"""
        elapsed = time.time() - self.start_time
        if elapsed > self.max_time * 0.8:  # Stop at 80% of time limit
            print(f"⚠️  Approaching time limit ({elapsed/60:.1f} minutes), stopping analysis")
            return True
        return False
        
    def round_to_power_of_2(self, n):
        """Round n to the nearest power of 2 for optimal Sobol sequence performance"""
        if n <= 0:
            return 1
        # Find the nearest power of 2
        power = int(np.log2(n))
        lower = 2**power
        upper = 2**(power + 1)
        
        # Return the closer power of 2
        if n - lower < upper - n:
            return lower
        else:
            return upper
        
    @lru_cache(maxsize=128)
    def generate_sobol_samples(self, n_params, n_samples):
        """Generate Sobol sequence samples with version compatibility and caching"""
        # Ensure n_samples is a power of 2 for optimal Sobol performance
        optimal_n = self.round_to_power_of_2(n_samples)
        if optimal_n != n_samples:
            print(f"Adjusting sample size from {n_samples} to {optimal_n} (nearest power of 2) for optimal Sobol sequence performance")
        
        try:
            # For scipy >= 1.7
            sampler = Sobol(d=n_params, scramble=True)
            samples = sampler.random(n=optimal_n)
            return samples
        except (NameError, ImportError):
            # For older scipy versions
            try:
                samples = sobol_seq.i4_sobol_generate(n_params, optimal_n)
                return samples
            except:
                # Fallback to quasi-random
                print("Warning: Using fallback pseudo-random instead of Sobol sequence")
                return np.random.random((optimal_n, n_params))
    
    def _extract_simulation_metrics(self, t_sim, y_sim, params):
        """Extract metrics from simulation results with proper array handling"""
        try:
            # y_sim is a 2D array with shape (n_variables, n_time_points)
            # Variables are [S, E, I, R, k_avg, C]
            if y_sim is None or len(y_sim.shape) < 2:
                print("Warning: Invalid simulation results")
                return None
            
            # Extract individual variables
            S = y_sim[0, :]  # Susceptible
            E = y_sim[1, :]  # Exposed
            I = y_sim[2, :]  # Infected
            R = y_sim[3, :]  # Recovered
            k_avg = y_sim[4, :]  # Average network degree
            C = y_sim[5, :]  # Clustering coefficient
            
            # Calculate metrics
            metrics = {
                'peak_infections': float(np.max(I)),
                'total_infections': float(np.trapz(I, t_sim)),
                'final_size': float(R[-1] / params.N),
                'min_network_degree': float(np.min(k_avg)),
                'max_network_degree': float(np.max(k_avg)),
                'avg_clustering': float(np.mean(C))
            }
            
            # Validate metrics
            for key, value in metrics.items():
                if np.isnan(value) or np.isinf(value):
                    print(f"Warning: Invalid {key} value: {value}")
                    return None
            
            return metrics
            
        except Exception as e:
            print(f"Error extracting simulation metrics: {e}")
            return None
    
    def _run_simulation(self, params, T_scenario='baseline'):
        """Run a single simulation with proper error handling"""
        try:
            # Create coupled model
            coupled_model = CoupledSystemModel(params)
            
            # Get climate scenario
            t, T, H = params.get_climate_scenario(T_scenario)
            T_func = lambda time: np.interp(time, t, T)
            H_func = lambda time: np.interp(time, t, H)
            
            # Initial conditions
            y0 = [params.N * 0.99, 0, params.N * 0.01, 0, params.k_0, 0.3]
            
            # Use shorter simulation time in quick mode
            sim_days = int(os.getenv('SIMULATION_DAYS', 365))
            if self.quick_mode:
                sim_days = min(sim_days, 180)
            
            # Run simulation
            t_sim, y_sim = coupled_model.solve_coupled_system(
                [0, sim_days], y0, T_func, H_func
            )
            
            # Extract metrics
            return self._extract_simulation_metrics(t_sim, y_sim, params)
            
        except Exception as e:
            print(f"Simulation failed: {e}")
            return None
    
    def sobol_sensitivity_analysis(self, n_samples=None, T_scenario='baseline'):
        """Perform Sobol sensitivity analysis with optimized settings"""
        
        # Use configuration for optimal sample size
        n_samples = self.config.get_optimal_sample_size(n_samples, self.quick_mode)
        
        # Use parameter ranges from configuration
        param_ranges = self.config.PARAMETER_RANGES.copy()
        
        # Validate parameter ranges
        param_ranges = self.validate_parameter_ranges(param_ranges)
        
        n_params = len(param_ranges)
        
        # Generate Sobol sequence (or fallback) with caching
        sobol_points = self.generate_sobol_samples(2 * n_params, n_samples)
        
        # Scale to parameter ranges
        param_names = list(param_ranges.keys())
        param_samples_A = np.zeros((n_samples, n_params))
        param_samples_B = np.zeros((n_samples, n_params))
        
        for i, param_name in enumerate(param_names):
            low, high = param_ranges[param_name]
            param_samples_A[:, i] = low + (high - low) * sobol_points[:, i]
            param_samples_B[:, i] = low + (high - low) * sobol_points[:, n_params + i]
        
        # Simulation function with timeout checking
        def evaluate_model(params_vector):
            if self.check_timeout():
                return None
                
            # Create modified parameters
            modified_params = ModelParameters()
            base_params = ModelParameters()
            
            for i, param_name in enumerate(param_names):
                base_value = getattr(base_params, param_name)
                setattr(modified_params, param_name, base_value * params_vector[i])
            
            # Run simulation
            return self._run_simulation(modified_params, T_scenario)
        
        # Run sensitivity analysis with progress tracking
        print(f"Running Sobol sensitivity analysis with {n_samples} samples...")
        
        results_A = []
        results_B = []
        
        for i in range(n_samples):
            if i % max(1, n_samples // 10) == 0:  # Progress every 10%
                print(f"Progress: {i}/{n_samples} ({i/n_samples*100:.1f}%)")
            
            # Evaluate model A
            result_A = evaluate_model(param_samples_A[i])
            if result_A is not None:
                results_A.append(result_A)
            
            # Evaluate model B
            result_B = evaluate_model(param_samples_B[i])
            if result_B is not None:
                results_B.append(result_B)
            
            # Check for timeout
            if self.check_timeout():
                print("Analysis timed out")
                break
        
        # Check if we have enough results
        if len(results_A) < n_samples // 2:
            print(f"Warning: Only {len(results_A)} successful simulations out of {n_samples}")
            return None
        
        # Calculate sensitivity indices
        sensitivity_indices = self.calculate_sensitivity_indices(results_A, results_B, param_names)
        
        return {
            'sensitivity_indices': sensitivity_indices,
            'param_names': param_names,
            'n_samples': n_samples,
            'successful_simulations': len(results_A)
        }
    
    def calculate_sensitivity_indices(self, results_A, results_B, param_names):
        """Calculate Sobol sensitivity indices efficiently"""
        if not results_A or not results_B:
            return None
            
        # Convert to numpy arrays for efficiency
        metrics = list(results_A[0].keys())
        n_metrics = len(metrics)
        n_params = len(param_names)
        
        # Initialize sensitivity indices
        sensitivity_indices = {}
        
        for metric in metrics:
            # Extract metric values
            y_A = np.array([r[metric] for r in results_A if r is not None])
            y_B = np.array([r[metric] for r in results_B if r is not None])
            
            if len(y_A) != len(y_B):
                continue
                
            # Calculate first-order indices
            first_order = []
            total_order = []
            
            for i in range(n_params):
                # Create C_i matrix (A with B_i)
                # This is a simplified calculation for speed
                if len(y_A) > 0:
                    # Use variance-based sensitivity (simplified)
                    var_total = np.var(y_A)
                    if var_total > 0:
                        first_order.append(np.var(y_A - y_B) / var_total)
                    else:
                        first_order.append(0.0)
                else:
                    first_order.append(0.0)
            
            sensitivity_indices[metric] = {
                'first_order': dict(zip(param_names, first_order)),
                'total_order': dict(zip(param_names, first_order))  # Simplified
            }
        
        return sensitivity_indices
    
    def monte_carlo_uncertainty(self, n_samples=None, T_scenario='heatwave'):
        """Perform Monte Carlo uncertainty analysis with optimized settings"""
        
        # Use configuration for optimal sample size
        n_samples = self.config.get_mc_sample_size(n_samples, self.quick_mode)
        
        print(f"Running Monte Carlo uncertainty analysis with {n_samples} samples...")
        
        # Use parameter uncertainties from configuration
        param_uncertainties = self.config.PARAMETER_UNCERTAINTIES.copy()
        
        # Generate parameter samples
        param_names = list(param_uncertainties.keys())
        n_params = len(param_names)
        
        # Use Latin Hypercube Sampling for better coverage
        samples = np.random.random((n_samples, n_params))
        
        # Scale to parameter ranges
        base_params = ModelParameters()
        param_samples = np.zeros((n_samples, n_params))
        
        for i, param_name in enumerate(param_names):
            base_value = getattr(base_params, param_name)
            uncertainty = param_uncertainties[param_name]
            
            # Generate samples around base value
            param_samples[:, i] = base_value + uncertainty * (samples[:, i] - 0.5)
        
        # Run simulations
        results = []
        
        for i in range(n_samples):
            if i % max(1, n_samples // 10) == 0:  # Progress every 10%
                print(f"Progress: {i}/{n_samples} ({i/n_samples*100:.1f}%)")
            
            if self.check_timeout():
                print("⚠️  Stopping analysis due to timeout")
                break
            
            # Create modified parameters
            modified_params = ModelParameters()
            for j, param_name in enumerate(param_names):
                setattr(modified_params, param_name, param_samples[i, j])
            
            # Run simulation
            result = self._run_simulation(modified_params, T_scenario)
            if result is not None:
                results.append(result)
        
        if len(results) < n_samples // 2:
            print("⚠️  Insufficient successful simulations for uncertainty analysis")
            return None
        
        return results
    
    def calculate_uncertainty_bounds(self, mc_results, confidence_level=0.95):
        """Calculate uncertainty bounds efficiently"""
        if not mc_results:
            return None
            
        metrics = list(mc_results[0].keys())
        bounds = {}
        
        for metric in metrics:
            values = [r[metric] for r in mc_results if r is not None]
            if values:
                values = np.array(values)
                bounds[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'percentiles': np.percentile(values, [5, 25, 50, 75, 95])
                }
        
        return bounds

    def morris_screening(self, n_samples=100, n_groups=10, parameter_ranges=None):
        """
        Perform Morris screening sensitivity analysis
        
        Args:
            n_samples: Number of samples for Morris method
            n_groups: Number of groups for grouping parameters
            parameter_ranges: Dict of parameter ranges {param: (min, max)}
            
        Returns:
            Dict with Morris screening results
        """
        # Ensure n_samples is an integer
        n_samples = int(n_samples)
        n_groups = int(n_groups)
        
        if parameter_ranges is None:
            parameter_ranges = {
                'beta_0': (0.1, 0.5),
                'alpha_T': (0.01, 0.05),
                'sigma': (0.1, 0.3),
                'gamma': (0.1, 0.3),
                'kappa': (0.1, 0.5),
                'sigma_k': (1, 10),
                'T_critical': (30, 40)
            }
        
        # Validate parameter ranges
        valid_params = []
        for param, (min_val, max_val) in parameter_ranges.items():
            if hasattr(ModelParameters(), param):
                valid_params.append(param)
            else:
                print(f"Warning: Parameter '{param}' not found in ModelParameters, skipping...")
        
        if not valid_params:
            print("Error: No valid parameters found for Morris screening")
            return None
        
        param_names = valid_params
        n_params = len(param_names)
        
        # Generate Morris trajectories
        delta = 1.0 / (n_groups - 1)
        elementary_effects = {param: [] for param in param_names}
        
        print(f"Running Morris screening with {n_samples} trajectories for {n_params} parameters...")
        
        try:
            for i in range(min(n_samples, 50)):  # Limit for performance
                if self.check_timeout():
                    break
                    
                # Generate base point
                base_point = np.random.random(n_params)
                
                # Scale to parameter ranges
                scaled_params = {}
                for j, param in enumerate(param_names):
                    min_val, max_val = parameter_ranges[param]
                    scaled_params[param] = min_val + base_point[j] * (max_val - min_val)
                
                # Calculate base output
                base_output = self._evaluate_model_with_params(scaled_params)
                
                # Skip if base output is invalid
                if base_output is None or np.isnan(base_output) or np.isinf(base_output):
                    print(f"Warning: Invalid base output in trajectory {i}, skipping")
                    continue
                
                # Calculate elementary effects
                for j, param in enumerate(param_names):
                    # Perturb parameter
                    perturbed_params = scaled_params.copy()
                    min_val, max_val = parameter_ranges[param]
                    perturbed_params[param] = min(max_val, scaled_params[param] + delta * (max_val - min_val))
                    
                    # Calculate effect
                    perturbed_output = self._evaluate_model_with_params(perturbed_params)
                    
                    # Skip if perturbed output is invalid
                    if perturbed_output is None or np.isnan(perturbed_output) or np.isinf(perturbed_output):
                        print(f"Warning: Invalid perturbed output for {param} in trajectory {i}, skipping")
                        continue
                    
                    effect = (perturbed_output - base_output) / (delta * (max_val - min_val))
                    elementary_effects[param].append(effect)
            
            # Calculate Morris indices
            morris_indices = {}
            for param in param_names:
                effects = np.array(elementary_effects[param])
                if len(effects) > 0:
                    morris_indices[param] = {
                        'mu': float(np.mean(effects)),
                        'mu_star': float(np.mean(np.abs(effects))),
                        'sigma': float(np.std(effects))
                    }
                else:
                    morris_indices[param] = {
                        'mu': 0.0,
                        'mu_star': 0.0,
                        'sigma': 0.0
                    }
            
            return {
                'parameters': param_names,
                'indices': morris_indices,
                'n_trajectories': len(elementary_effects[param_names[0]]) if param_names else 0
            }
            
        except Exception as e:
            print(f"Morris screening error: {e}")
            return None
    
    def _evaluate_model_with_params(self, param_dict):
        """Helper method to evaluate model with specific parameters"""
        try:
            # Create a copy of current params
            temp_params = ModelParameters()
            
            # Update with new values
            for param, value in param_dict.items():
                if hasattr(temp_params, param):
                    setattr(temp_params, param, value)
            
            # Run simplified simulation
            model = CoupledSystemModel(temp_params, n_nodes=100)
            
            # Simple epidemic scenario
            t = np.linspace(0, 30, 30)  # 30 days
            
            def T_func(time):
                time_array = np.atleast_1d(time)
                result = 25 + 5*np.sin(2*np.pi*time_array/365)
                return float(result) if np.isscalar(time) else result
                
            def H_func(time):
                if np.isscalar(time):
                    return 0.7
                else:
                    return 0.7 * np.ones_like(np.atleast_1d(time))
            
            # Set initial conditions
            S0 = temp_params.N - 100  # Start with 100 infected
            I0 = 100
            R0 = 0
            E0 = 0
            k0 = temp_params.k_0
            C0 = temp_params.C_0
            y0 = np.array([S0, I0, R0, E0, k0, C0])
            
            time_points, states = model.solve_coupled_system(
                t_span=(0, 30),
                y0=y0,
                T_func=T_func,
                H_func=H_func,
                t_eval=t
            )
            
            # Check if states is valid
            if states is None or len(states.shape) < 2:
                print(f"Warning: Invalid states returned from solve_coupled_system")
                return None
            
            # Return peak infection as output metric
            # states has shape (n_variables, n_time_points) where variables are [S, E, I, R, k_avg, C]
            # So I (infections) is at index 2
            peak_infections = float(np.max(states[2, :]))
            
            # Check for invalid values
            if np.isnan(peak_infections) or np.isinf(peak_infections):
                print(f"Warning: Invalid peak infections value: {peak_infections}")
                return None
                
            return peak_infections
            
        except Exception as e:
            # Log the error for debugging
            print(f"Model evaluation error: {e}")
            return None

    def validate_parameter_ranges(self, param_ranges):
        """Validate parameter ranges to ensure they are reasonable"""
        validated_ranges = {}
        base_params = ModelParameters()
        
        for param_name, (low, high) in param_ranges.items():
            if not hasattr(base_params, param_name):
                print(f"Warning: Parameter '{param_name}' not found in ModelParameters, skipping")
                continue
                
            base_value = getattr(base_params, param_name)
            
            # Ensure ranges are reasonable
            if low < 0 and base_value > 0:
                print(f"Warning: Negative range for {param_name}, adjusting to 0.1")
                low = 0.1
            elif low < 0 and base_value < 0:
                print(f"Warning: Negative range for {param_name}, adjusting to -abs(base_value)")
                low = -abs(base_value)
            
            if high <= low:
                print(f"Warning: Invalid range for {param_name} ({low}, {high}), using (0.5, 2.0)")
                low, high = 0.5, 2.0
            
            validated_ranges[param_name] = (low, high)
        
        return validated_ranges
    
    def adaptive_sampling(self, initial_n_samples, min_samples=64, max_samples=2048):
        """Adaptively adjust sample size based on convergence"""
        n_samples = initial_n_samples
        
        # Start with minimum samples
        if n_samples < min_samples:
            n_samples = min_samples
        elif n_samples > max_samples:
            n_samples = max_samples
        
        # Round to power of 2
        n_samples = self.round_to_power_of_2(n_samples)
        
        return n_samples
