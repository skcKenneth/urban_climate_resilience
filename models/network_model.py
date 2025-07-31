"""
Dynamic social network evolution model
"""
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from utils.parameters import ModelParameters

class DynamicNetworkModel:
    """Social network with climate-dependent evolution"""
    
    def __init__(self, params=None, n_nodes=1000):
        self.params = params if params else ModelParameters()
        self.n_nodes = n_nodes
        self.positions = self._generate_positions()
        self.distance_matrix = self._compute_distances()
        
    def _generate_positions(self):
        """Generate random geographic positions for nodes"""
        return np.random.uniform(0, 10, (self.n_nodes, 2))
    
    def _compute_distances(self):
        """Compute distance matrix between all nodes"""
        pos = self.positions
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        return distances
    
    def climate_stress_function(self, T):
        """Network formation probability under climate stress"""
        return np.exp(-(T - self.params.T_opt)**2 / (2 * self.params.sigma_T**2))
    
    def edge_dissolution_rate(self, T):
        """Rate of edge dissolution under temperature stress"""
        if T <= self.params.T_threshold:
            return self.params.h_0
        return self.params.h_0 * (
            1 + np.exp((T - self.params.T_threshold) / self.params.delta_T)
        )
    
    def connection_probability(self, i, j, degrees, T, total_edges):
        """Probability of forming connection between nodes i and j"""
        if i == j:
            return 0
        
        # Preferential attachment
        pref_attach = degrees[i] * degrees[j] / (2 * max(total_edges, 1))
        
        # Geographic proximity
        geographic = np.exp(-self.params.lambda_dist * self.distance_matrix[i, j])
        
        # Climate stress
        climate = self.climate_stress_function(T)
        
        return pref_attach * geographic * climate
    
    def evolve_network(self, G, T, dt):
        """Evolve network for one time step"""
        n = len(G.nodes())
        degrees = dict(G.degree())
        total_edges = G.number_of_edges()
        
        # Edge formation
        for i in range(n):
            for j in range(i+1, n):
                if not G.has_edge(i, j):
                    prob = self.connection_probability(
                        i, j, [degrees[i], degrees[j]], T, total_edges
                    ) * dt
                    if np.random.random() < prob:
                        G.add_edge(i, j)
                        degrees[i] += 1
                        degrees[j] += 1
                        total_edges += 1
        
        # Edge dissolution
        dissolution_rate = self.edge_dissolution_rate(T) * dt
        edges_to_remove = []
        for edge in G.edges():
            if np.random.random() < dissolution_rate:
                edges_to_remove.append(edge)
        
        G.remove_edges_from(edges_to_remove)
        return G
    
    def initialize_network(self, initial_degree=None):
        """Create initial network configuration"""
        if initial_degree is None:
            initial_degree = self.params.k_0
        
        # Create scale-free network
        G = nx.barabasi_albert_graph(self.n_nodes, int(initial_degree/2))
        
        # Add geographic constraints
        edges_to_remove = []
        for edge in G.edges():
            i, j = edge
            dist = self.distance_matrix[i, j]
            if dist > 5.0:  # Remove long-distance connections
                edges_to_remove.append(edge)
        
        G.remove_edges_from(edges_to_remove)
        return G
    
    def network_statistics(self, G):
        """Compute network statistics"""
        if G.number_of_edges() == 0:
            return {
                'avg_degree': 0,
                'clustering': 0,
                'density': 0,
                'components': 1
            }
        
        stats = {
            'avg_degree': np.mean([d for n, d in G.degree()]),
            'clustering': nx.average_clustering(G),
            'density': nx.density(G),
            'components': nx.number_connected_components(G)
        }
        
        return stats
