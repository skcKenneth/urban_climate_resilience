"""
Synthetic data generation for model testing and validation
"""
import numpy as np
import networkx as nx
from utils.parameters import ModelParameters

class DataGenerator:
    """Generate synthetic data for testing and validation"""
    
    def __init__(self, params=None, seed=42):
        self.params = params if params else ModelParameters()
        self.rng = np.random.RandomState(seed)
        
    def generate_climate_scenarios(self):
        """Generate multiple climate scenarios"""
        scenarios = {}
        
        # Baseline scenario
        t_baseline, T_baseline, H_baseline = self.params.get_climate_scenario('baseline')
        scenarios['baseline'] = {'t': t_baseline, 'T': T_baseline, 'H': H_baseline}
        
        # Heatwave scenario
        t_heatwave, T_heatwave, H_heatwave = self.params.get_climate_scenario('heatwave')
        scenarios['heatwave'] = {'t': t_heatwave, 'T': T_heatwave, 'H': H_heatwave}
        
        # Extreme scenario (multiple stressors)
        t = np.arange(0, self.params.T_sim, self.params.dt)
        T_extreme = 28 + 12*np.sin(2*np.pi*t/365) + 5*self.rng.normal(0, 1, len(t))
        
        # Add multiple heatwave events
        heatwave_times = [80, 150, 220, 290]
        for hw_time in heatwave_times:
            mask = (t >= hw_time) & (t <= hw_time + 10)
            T_extreme[mask] += 10*np.exp(-(t[mask] - hw_time - 5)**2/15)
        
        H_extreme = 0.7 + 0.3*np.sin(2*np.pi*t/365 + np.pi/3) + 0.1*self.rng.normal(0, 1, len(t))
        H_extreme = np.clip(H_extreme, 0.2, 0.9)
        
        scenarios['extreme'] = {'t': t, 'T': T_extreme, 'H': H_extreme}
        
        return scenarios
    
    def generate_climate_scenario(self, scenario_type='baseline', days=365):
        """Generate a single climate scenario"""
        t = np.linspace(0, days, days)
        
        if scenario_type == 'baseline':
            T = 20 + 10*np.sin(2*np.pi*t/365)
            H = 0.6 + 0.2*np.sin(2*np.pi*t/365 + np.pi/4)
        elif scenario_type == 'heatwave':
            T = 25 + 10*np.sin(2*np.pi*t/365)
            # Add heatwave event
            heatwave_start = days // 2
            mask = (t >= heatwave_start) & (t <= heatwave_start + 10)
            T[mask] += 8*np.exp(-(t[mask] - heatwave_start - 5)**2/10)
            H = 0.7 + 0.2*np.sin(2*np.pi*t/365 + np.pi/4)
        elif scenario_type == 'extreme':
            T = 28 + 12*np.sin(2*np.pi*t/365) + 2*self.rng.normal(0, 1, len(t))
            # Add multiple heatwave events
            heatwave_times = [days//4, days//2, 3*days//4]
            for hw_time in heatwave_times:
                mask = (t >= hw_time) & (t <= hw_time + 10)
                T[mask] += 10*np.exp(-(t[mask] - hw_time - 5)**2/15)
            H = 0.8 + 0.15*np.sin(2*np.pi*t/365 + np.pi/3)
        else:
            # Default to baseline
            T = 20 + 10*np.sin(2*np.pi*t/365)
            H = 0.6 + 0.2*np.sin(2*np.pi*t/365 + np.pi/4)
            
        return T, H
    
    def temperature_function(self):
        """Return a temperature function for use in models"""
        def T_func(t):
            return 20 + 10*np.sin(2*np.pi*t/365) + 2*np.random.normal(0, 1)
        return T_func
    
    def humidity_function(self):
        """Return a humidity function for use in models"""
        def H_func(t):
            return 0.6 + 0.2*np.sin(2*np.pi*t/365 + np.pi/4)
        return H_func
    
    def generate_initial_conditions(self, n_scenarios=5):
        """Generate different initial condition scenarios"""
        scenarios = {}
        
        # Base case
        scenarios['base'] = [
            self.params.N * 0.99,  # S
            0,                      # E  
            self.params.N * 0.01,  # I
            0,                      # R
            self.params.k_0,       # k_avg
            0.3                     # C
        ]
        
        # Multiple introduction points
        scenarios['multiple_seeds'] = [
            self.params.N * 0.98,
            self.params.N * 0.01,
            self.params.N * 0.01,
            0,
            self.params.k_0,
            0.3
        ]
        
        # Fragmented network start
        scenarios['fragmented_network'] = [
            self.params.N * 0.99,
            0,
            self.params.N * 0.01,
            0,
            self.params.k_0 * 0.6,  # Lower initial connectivity
            0.15                     # Lower clustering
        ]
        
        # High clustering start
        scenarios['high_clustering'] = [
            self.params.N * 0.99,
            0,
            self.params.N * 0.01,
            0,
            self.params.k_0 * 0.8,
            0.5                     # High clustering
        ]
        
        # Random variations
        for i in range(n_scenarios):
            I_frac = self.rng.uniform(0.005, 0.02)
            k_mult = self.rng.uniform(0.7, 1.3)
            C_val = self.rng.uniform(0.15, 0.45)
            
            scenarios[f'random_{i}'] = [
                self.params.N * (1 - I_frac),
                0,
                self.params.N * I_frac,
                0,
                self.params.k_0 * k_mult,
                C_val
            ]
        
        return scenarios
    
    def generate_control_scenarios(self):
        """Generate different control policy scenarios"""
        scenarios = {}
        
        # No control
        scenarios['no_control'] = lambda t: [0, 0, 0]
        
        # Constant controls
        scenarios['low_constant'] = lambda t: [0.3, 0.3, 0.3]
        scenarios['high_constant'] = lambda t: [0.8, 0.8, 0.8]
        
        # Step functions
        def step_control(switch_time, before, after):
            def control_func(t):
                if t < switch_time:
                    return before
                else:
                    return after
            return control_func
        
        scenarios['early_intervention'] = step_control(50, [0.8, 0.5, 0.3], [0.2, 0.2, 0.1])
        scenarios['late_intervention'] = step_control(150, [0.0, 0.0, 0.0], [0.8, 0.8, 0.5])
        
        # Reactive control (temperature-based)
        def reactive_control(T_func, thresholds=[30, 35]):
            def control_func(t):
                T = T_func(t)
                if T < thresholds[0]:
                    return [0.1, 0.1, 0.0]
                elif T < thresholds[1]:
                    return [0.5, 0.4, 0.3]
                else:
                    return [0.9, 0.8, 0.7]
            return control_func
        
        scenarios['reactive'] = reactive_control
        
        # Periodic control
        def periodic_control(period=30, intensity=0.6):
            def control_func(t):
                phase = (t % period) / period
                if phase < 0.5:
                    return [intensity, intensity*0.7, intensity*0.5]
                else:
                    return [0.1, 0.1, 0.0]
            return control_func
        
        scenarios['periodic'] = periodic_control()
        
        return scenarios
    
    def generate_validation_data(self, n_points=50):
        """Generate synthetic data for model validation"""
        
        # Create parameter variations
        param_variations = {
            'beta_0': np.linspace(0.1, 0.6, n_points),
            'sigma': np.linspace(0.1, 0.4, n_points),
            'gamma': np.linspace(0.05, 0.3, n_points),
            'alpha_T': np.linspace(0.01, 0.05, n_points),
            'kappa': np.linspace(0.2, 0.8, n_points)
        }
        
        validation_data = []
        
        for i in range(n_points):
            # Create parameter set
            params = ModelParameters()
            for param_name in param_variations:
                value_idx = self.rng.randint(0, len(param_variations[param_name]))
                setattr(params, param_name, param_variations[param_name][value_idx])
            
            # Generate climate scenario
            t, T, H = params.get_climate_scenario('baseline')
            
            # Add observation noise
            obs_noise_T = self.rng.normal(0, 0.5, len(T))
            obs_noise_H = self.rng.normal(0, 0.02, len(H))
            
            T_observed = T + obs_noise_T
            H_observed = np.clip(H + obs_noise_H, 0, 1)
            
            validation_data.append({
                'parameters': {name: getattr(params, name) for name in param_variations},
                'climate': {
                    't': t,
                    'T_true': T,
                    'H_true': H,
                    'T_observed': T_observed,
                    'H_observed': H_observed
                },
                'initial_conditions': [
                    params.N * self.rng.uniform(0.98, 1.0),
                    0,
                    params.N * self.rng.uniform(0.005, 0.02),
                    0,
                    params.k_0 * self.rng.uniform(0.8, 1.2),
                    self.rng.uniform(0.2, 0.4)
                ]
            })
        
        return validation_data
    
    def generate_network_snapshots(self, n_nodes=200, n_snapshots=10):
        """Generate network evolution snapshots"""
        
        # Initial network
        G = nx.barabasi_albert_graph(n_nodes, 3)
        
        # Generate geographic positions
        pos = {i: self.rng.uniform(0, 10, 2) for i in range(n_nodes)}
        nx.set_node_attributes(G, pos, 'pos')
        
        snapshots = []
        
        # Temperature scenario
        T_values = np.linspace(20, 40, n_snapshots)
        
        for i, T in enumerate(T_values):
            # Calculate network statistics
            stats = {
                'time': i * 30,  # 30 days between snapshots
                'temperature': T,
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'avg_degree': np.mean([d for n, d in G.degree()]),
                'clustering': nx.average_clustering(G),
                'density': nx.density(G),
                'components': nx.number_connected_components(G)
            }
            
            snapshots.append({
                'network': G.copy(),
                'statistics': stats
            })
            
            # Evolve network based on temperature
            if T > 30:  # High temperature - remove edges
                edges_to_remove = []
                dissolution_prob = 0.1 * (T - 30) / 10
                
                for edge in G.edges():
                    if self.rng.random() < dissolution_prob:
                        edges_to_remove.append(edge)
                
                G.remove_edges_from(edges_to_remove)
            
            # Add some new edges (recovery/adaptation)
            if i > 2 and T < 35:
                n_new_edges = max(1, int(0.05 * G.number_of_nodes()))
                nodes = list(G.nodes())
                
                for _ in range(n_new_edges):
                    u, v = self.rng.choice(nodes, 2, replace=False)
                    if not G.has_edge(u, v):
                        # Distance-based connection probability
                        pos_u = pos[u]
                        pos_v = pos[v]
                        distance = np.sqrt(sum((pos_u[i] - pos_v[i])**2 for i in range(2)))
                        if distance < 3.0:  # Only connect nearby nodes
                            G.add_edge(u, v)
        
        return snapshots
