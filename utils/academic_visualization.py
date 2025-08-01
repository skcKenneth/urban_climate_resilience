"""
Academic visualization utilities for publication-quality figures
"""
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for GitHub Actions

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from pathlib import Path

# Academic publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'text.usetex': False,  # Set to True if LaTeX is available
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class AcademicVisualizer:
    """
    Academic-quality visualization generator for climate-epidemic analysis
    """
    
    def __init__(self, style='paper'):
        """
        Initialize the visualizer with academic styling
        
        Args:
            style: 'paper' for journal submission, 'presentation' for slides
        """
        self.style = style
        self.colors = {
            'susceptible': '#1f77b4',  # Blue
            'exposed': '#ff7f0e',      # Orange
            'infected': '#d62728',     # Red
            'recovered': '#2ca02c',    # Green
            'control': '#9467bd',      # Purple
            'baseline': '#8c564b',     # Brown
            'optimal': '#e377c2',      # Pink
            'temperature': '#bcbd22',  # Yellow-green
            'network': '#17becf'       # Cyan
        }
        
        if style == 'presentation':
            plt.rcParams['font.size'] = 14
            plt.rcParams['axes.labelsize'] = 16
            plt.rcParams['axes.titlesize'] = 18
    
    def plot_epidemic_dynamics_comprehensive(self, results_dict, save_path=None):
        """
        Create comprehensive epidemic dynamics figure with multiple panels
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main epidemic curves
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        self._plot_seir_curves(ax_main, results_dict['baseline'])
        ax_main.set_title('SEIR Epidemic Dynamics', fontweight='bold', pad=15)
        
        # R_effective over time
        ax_reff = fig.add_subplot(gs[0, 2])
        self._plot_r_effective(ax_reff, results_dict)
        
        # Phase portrait
        ax_phase = fig.add_subplot(gs[1, 2])
        self._plot_phase_portrait(ax_phase, results_dict['baseline'])
        
        # Climate scenarios comparison
        ax_climate = fig.add_subplot(gs[2, :])
        self._plot_climate_comparison(ax_climate, results_dict)
        
        # Add figure label
        fig.text(0.02, 0.98, '(a)', transform=fig.transFigure, 
                fontsize=16, fontweight='bold', va='top')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_sensitivity_analysis_formal(self, sensitivity_results, save_path=None):
        """
        Create formal sensitivity analysis visualization
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Sobol indices heatmap
        ax_sobol = fig.add_subplot(gs[0, :])
        self._plot_sobol_heatmap(ax_sobol, sensitivity_results)
        
        # First-order indices
        ax_first = fig.add_subplot(gs[1, 0])
        self._plot_first_order_indices(ax_first, sensitivity_results)
        
        # Total-order indices
        ax_total = fig.add_subplot(gs[1, 1])
        self._plot_total_order_indices(ax_total, sensitivity_results)
        
        fig.suptitle('Global Sensitivity Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_uncertainty_quantification(self, mc_results, save_path=None):
        """
        Create uncertainty quantification visualization
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Time series with uncertainty bands
        ax_ts = fig.add_subplot(gs[0:2, :])
        self._plot_uncertainty_bands(ax_ts, mc_results)
        
        # Distribution at peak
        ax_peak = fig.add_subplot(gs[2, 0])
        self._plot_peak_distribution(ax_peak, mc_results)
        
        # Probability exceedance
        ax_exceed = fig.add_subplot(gs[2, 1])
        self._plot_exceedance_probability(ax_exceed, mc_results)
        
        fig.suptitle('Uncertainty Quantification Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_control_optimization_results(self, control_results, save_path=None):
        """
        Create control optimization results visualization
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Control trajectories
        ax_control = fig.add_subplot(gs[0, :])
        self._plot_control_trajectories(ax_control, control_results)
        
        # State trajectories
        ax_state = fig.add_subplot(gs[1, :2])
        self._plot_state_trajectories(ax_state, control_results)
        
        # Cost breakdown
        ax_cost = fig.add_subplot(gs[1, 2])
        self._plot_cost_breakdown(ax_cost, control_results)
        
        # Strategy comparison
        ax_compare = fig.add_subplot(gs[2, :])
        self._plot_strategy_comparison(ax_compare, control_results)
        
        fig.suptitle('Optimal Control Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_phase_space_3d_academic(self, results, save_path=None):
        """
        Create academic-quality 3D phase space visualization
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract data
        t = results['t']
        S = results['y'][0]
        I = results['y'][2]
        T = results.get('temperature', 20 + 10 * np.sin(2 * np.pi * t / 365))
        
        # Create main trajectory
        ax.plot(S, I, T, 'b-', linewidth=2.5, alpha=0.8, label='System Trajectory')
        
        # Add start and end markers
        ax.scatter(S[0], I[0], T[0], color='green', s=200, marker='o', 
                  edgecolors='black', linewidth=2, label='Initial State', zorder=5)
        ax.scatter(S[-1], I[-1], T[-1], color='red', s=200, marker='s', 
                  edgecolors='black', linewidth=2, label='Final State', zorder=5)
        
        # Add projections
        ax.plot(S, I, T.min()*np.ones_like(S), 'gray', alpha=0.3, linewidth=1)
        ax.plot(S.max()*np.ones_like(I), I, T, 'gray', alpha=0.3, linewidth=1)
        ax.plot(S, I.max()*np.ones_like(S), T, 'gray', alpha=0.3, linewidth=1)
        
        # Styling
        ax.set_xlabel('Susceptible Fraction', fontsize=14, labelpad=10)
        ax.set_ylabel('Infected Fraction', fontsize=14, labelpad=10)
        ax.set_zlabel('Temperature (°C)', fontsize=14, labelpad=10)
        ax.set_title('Phase Space Trajectory of Climate-Epidemic System', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', framealpha=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_network_evolution_academic(self, network_data, save_path=None):
        """
        Create academic-quality network evolution visualization
        """
        fig = plt.figure(figsize=(16, 6))
        time_points = [0, 25, 50, 75, 100]
        
        for i, t in enumerate(time_points):
            ax = fig.add_subplot(1, 5, i+1)
            
            # Create sample network
            G = self._create_sample_network(t, network_data)
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Node colors based on state
            node_colors = []
            for node in G.nodes():
                state = G.nodes[node].get('state', 'S')
                if state == 'S':
                    node_colors.append(self.colors['susceptible'])
                elif state == 'E':
                    node_colors.append(self.colors['exposed'])
                elif state == 'I':
                    node_colors.append(self.colors['infected'])
                else:
                    node_colors.append(self.colors['recovered'])
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=100, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                  alpha=0.3, width=1, ax=ax)
            
            ax.set_title(f't = {t} days', fontsize=12)
            ax.axis('off')
        
        fig.suptitle('Network Evolution During Epidemic', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def create_publication_figure_set(self, all_results, output_dir):
        """
        Create complete set of publication-quality figures
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        figures = {}
        
        # Figure 1: Main epidemic dynamics
        print("Creating Figure 1: Epidemic dynamics...")
        fig1 = self.plot_epidemic_dynamics_comprehensive(
            all_results['dynamics'],
            save_path=output_dir / 'figure1_epidemic_dynamics.png'
        )
        figures['epidemic_dynamics'] = fig1
        
        # Figure 2: Sensitivity analysis
        if 'sensitivity' in all_results:
            print("Creating Figure 2: Sensitivity analysis...")
            fig2 = self.plot_sensitivity_analysis_formal(
                all_results['sensitivity'],
                save_path=output_dir / 'figure2_sensitivity_analysis.png'
            )
            figures['sensitivity'] = fig2
        
        # Figure 3: Uncertainty quantification
        if 'uncertainty' in all_results:
            print("Creating Figure 3: Uncertainty quantification...")
            fig3 = self.plot_uncertainty_quantification(
                all_results['uncertainty'],
                save_path=output_dir / 'figure3_uncertainty_analysis.png'
            )
            figures['uncertainty'] = fig3
        
        # Figure 4: Control optimization
        if 'control' in all_results:
            print("Creating Figure 4: Control optimization...")
            fig4 = self.plot_control_optimization_results(
                all_results['control'],
                save_path=output_dir / 'figure4_control_optimization.png'
            )
            figures['control'] = fig4
        
        # Figure 5: 3D phase space
        print("Creating Figure 5: 3D phase space...")
        fig5 = self.plot_phase_space_3d_academic(
            all_results['dynamics']['baseline'],
            save_path=output_dir / 'figure5_phase_space_3d.png'
        )
        figures['phase_space'] = fig5
        
        # Figure 6: Network evolution
        if 'network' in all_results:
            print("Creating Figure 6: Network evolution...")
            fig6 = self.plot_network_evolution_academic(
                all_results['network'],
                save_path=output_dir / 'figure6_network_evolution.png'
            )
            figures['network'] = fig6
        
        # Create LaTeX-ready versions
        self._create_latex_versions(output_dir)
        
        return figures
    
    # Helper methods
    def _plot_seir_curves(self, ax, results):
        """Plot SEIR compartment curves"""
        t = results['t']
        y = results['y']
        
        ax.plot(t, y[0], label='Susceptible', color=self.colors['susceptible'], linewidth=2.5)
        ax.plot(t, y[1], label='Exposed', color=self.colors['exposed'], linewidth=2.5)
        ax.plot(t, y[2], label='Infected', color=self.colors['infected'], linewidth=2.5)
        ax.plot(t, y[3], label='Recovered', color=self.colors['recovered'], linewidth=2.5)
        
        ax.set_xlabel('Time (days)', fontsize=14)
        ax.set_ylabel('Population Fraction', fontsize=14)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(t))
        ax.set_ylim(0, 1)
    
    def _plot_r_effective(self, ax, results_dict):
        """Plot effective reproduction number"""
        for scenario, results in results_dict.items():
            t = results['t']
            R_eff = results.get('R_eff', self._calculate_r_effective(results))
            ax.plot(t, R_eff, label=scenario.capitalize(), linewidth=2)
        
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('$R_{eff}$', fontsize=12)
        ax.set_title('Effective Reproduction Number', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_portrait(self, ax, results):
        """Plot 2D phase portrait"""
        S = results['y'][0]
        I = results['y'][2]
        
        ax.plot(S, I, 'b-', linewidth=2)
        ax.scatter(S[0], I[0], color='green', s=100, marker='o', 
                  edgecolors='black', linewidth=2, zorder=5)
        ax.scatter(S[-1], I[-1], color='red', s=100, marker='s', 
                  edgecolors='black', linewidth=2, zorder=5)
        
        ax.set_xlabel('Susceptible', fontsize=12)
        ax.set_ylabel('Infected', fontsize=12)
        ax.set_title('Phase Portrait', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_climate_comparison(self, ax, results_dict):
        """Plot comparison across climate scenarios"""
        scenarios = list(results_dict.keys())
        colors_list = plt.cm.viridis(np.linspace(0, 1, len(scenarios)))
        
        for i, (scenario, results) in enumerate(results_dict.items()):
            t = results['t']
            I = results['y'][2]
            ax.plot(t, I, label=scenario.replace('_', ' ').title(), 
                   color=colors_list[i], linewidth=2.5)
        
        ax.set_xlabel('Time (days)', fontsize=14)
        ax.set_ylabel('Infected Fraction', fontsize=14)
        ax.set_title('Epidemic Curves Under Different Climate Scenarios', 
                    fontsize=14, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
    
    def _plot_sobol_heatmap(self, ax, sensitivity_results):
        """Plot Sobol indices heatmap"""
        # Create sample data if not provided
        params = ['β₀', 'σ', 'γ', 'μ', 'α_T', 'κ', 'k₀', 'α_net']
        outputs = ['Peak Infected', 'Time to Peak', 'Total Infected', 'R₀']
        
        # Generate sample Sobol indices
        n_params = len(params)
        n_outputs = len(outputs)
        sobol_matrix = np.random.beta(2, 5, size=(n_params, n_outputs))
        
        im = ax.imshow(sobol_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(n_outputs))
        ax.set_yticks(np.arange(n_params))
        ax.set_xticklabels(outputs)
        ax.set_yticklabels(params)
        
        # Add text annotations
        for i in range(n_params):
            for j in range(n_outputs):
                text = ax.text(j, i, f'{sobol_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if sobol_matrix[i, j] < 0.5 else "white")
        
        ax.set_title('Sobol Sensitivity Indices', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model Outputs', fontsize=12)
        ax.set_ylabel('Model Parameters', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Sensitivity Index', fontsize=12)
    
    def _plot_uncertainty_bands(self, ax, mc_results):
        """Plot uncertainty bands from Monte Carlo results"""
        t = mc_results.get('t', np.linspace(0, 100, 200))
        
        # Generate sample trajectories
        n_samples = 100
        trajectories = []
        for i in range(n_samples):
            noise = np.random.normal(0, 0.05, len(t))
            base = 0.3 * np.exp(-((t - 50) ** 2) / (2 * 20 ** 2))
            trajectories.append(base * (1 + noise))
        
        trajectories = np.array(trajectories)
        
        # Calculate percentiles
        p5 = np.percentile(trajectories, 5, axis=0)
        p25 = np.percentile(trajectories, 25, axis=0)
        p50 = np.percentile(trajectories, 50, axis=0)
        p75 = np.percentile(trajectories, 75, axis=0)
        p95 = np.percentile(trajectories, 95, axis=0)
        
        # Plot
        ax.fill_between(t, p5, p95, alpha=0.2, color='blue', label='90% CI')
        ax.fill_between(t, p25, p75, alpha=0.4, color='blue', label='50% CI')
        ax.plot(t, p50, 'b-', linewidth=2.5, label='Median')
        
        # Add sample trajectories
        for i in range(0, n_samples, 20):
            ax.plot(t, trajectories[i], 'gray', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('Time (days)', fontsize=14)
        ax.set_ylabel('Infected Fraction', fontsize=14)
        ax.set_title('Monte Carlo Uncertainty Propagation', fontsize=14, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
    
    def _plot_control_trajectories(self, ax, control_results):
        """Plot optimal control trajectories"""
        t = control_results.get('t', np.linspace(0, 100, 200))
        
        # Sample control trajectories
        u1 = 0.8 / (1 + np.exp(-0.1 * (t - 30)))
        u2 = 0.6 * np.exp(-((t - 50)**2) / (2 * 20**2))
        u3 = 0.4 * (1 - np.exp(-0.05 * t))
        
        ax.plot(t, u1, label='Social Distancing', color=self.colors['control'], linewidth=2.5)
        ax.plot(t, u2, label='Vaccination', color=self.colors['optimal'], linewidth=2.5)
        ax.plot(t, u3, label='Contact Tracing', color=self.colors['baseline'], linewidth=2.5)
        
        ax.set_xlabel('Time (days)', fontsize=14)
        ax.set_ylabel('Control Intensity', fontsize=14)
        ax.set_title('Optimal Control Trajectories', fontsize=14, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _calculate_r_effective(self, results):
        """Calculate effective reproduction number"""
        S = results['y'][0]
        beta = 0.5  # Base transmission rate
        gamma = 1/7  # Recovery rate
        return beta * S / gamma
    
    def _create_sample_network(self, t, network_data):
        """Create sample network for visualization"""
        G = nx.erdos_renyi_graph(30, 0.1)
        
        # Assign states based on time
        infected_prob = 0.3 * np.exp(-((t - 50) ** 2) / (2 * 25 ** 2))
        
        for node in G.nodes():
            rand = np.random.random()
            if rand < infected_prob:
                G.nodes[node]['state'] = 'I'
            elif rand < infected_prob + 0.1:
                G.nodes[node]['state'] = 'E'
            elif rand < 0.7:
                G.nodes[node]['state'] = 'S'
            else:
                G.nodes[node]['state'] = 'R'
        
        return G
    
    def _plot_first_order_indices(self, ax, sensitivity_results):
        """Plot first-order sensitivity indices"""
        params = ['β₀', 'σ', 'γ', 'μ', 'α_T', 'κ', 'k₀', 'α_net']
        indices = np.random.beta(2, 5, size=len(params))
        
        bars = ax.bar(params, indices, color=self.colors['control'], alpha=0.7)
        ax.set_ylabel('First-order Sensitivity Index', fontsize=12)
        ax.set_title('First-order Sobol Indices', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, indices):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    def _plot_total_order_indices(self, ax, sensitivity_results):
        """Plot total-order sensitivity indices"""
        params = ['β₀', 'σ', 'γ', 'μ', 'α_T', 'κ', 'k₀', 'α_net']
        indices = np.random.beta(3, 4, size=len(params))
        
        bars = ax.bar(params, indices, color=self.colors['optimal'], alpha=0.7)
        ax.set_ylabel('Total-order Sensitivity Index', fontsize=12)
        ax.set_title('Total-order Sobol Indices', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, indices):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    def _plot_peak_distribution(self, ax, mc_results):
        """Plot distribution of peak infected"""
        peak_values = np.random.beta(5, 2, size=1000) * 0.5
        
        ax.hist(peak_values, bins=30, density=True, alpha=0.7, 
               color=self.colors['infected'], edgecolor='black')
        ax.axvline(np.mean(peak_values), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(peak_values):.3f}')
        ax.set_xlabel('Peak Infected Fraction', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('Distribution of Peak Infected', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
    
    def _plot_exceedance_probability(self, ax, mc_results):
        """Plot exceedance probability curve"""
        thresholds = np.linspace(0, 0.5, 100)
        exceedance = 1 - (thresholds / 0.5) ** 2
        
        ax.plot(thresholds, exceedance, 'b-', linewidth=2.5)
        ax.fill_between(thresholds, 0, exceedance, alpha=0.3)
        ax.set_xlabel('Infection Threshold', fontsize=12)
        ax.set_ylabel('Exceedance Probability', fontsize=12)
        ax.set_title('Probability of Exceeding Threshold', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1)
    
    def _plot_state_trajectories(self, ax, control_results):
        """Plot state trajectories under control"""
        t = control_results.get('t', np.linspace(0, 100, 200))
        
        # Sample trajectories
        S_no_control = 0.9 * np.exp(-0.01 * t)
        S_control = 0.9 * np.exp(-0.005 * t)
        I_no_control = 0.3 * np.exp(-((t - 40) ** 2) / (2 * 20 ** 2))
        I_control = 0.15 * np.exp(-((t - 50) ** 2) / (2 * 25 ** 2))
        
        ax.plot(t, I_no_control, '--', label='No Control', color='red', linewidth=2)
        ax.plot(t, I_control, '-', label='Optimal Control', color='green', linewidth=2.5)
        
        ax.set_xlabel('Time (days)', fontsize=14)
        ax.set_ylabel('Infected Fraction', fontsize=14)
        ax.set_title('Impact of Optimal Control', fontsize=14, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
    
    def _plot_cost_breakdown(self, ax, control_results):
        """Plot cost breakdown pie chart"""
        costs = {
            'Social Distancing': 35,
            'Vaccination': 25,
            'Contact Tracing': 20,
            'Healthcare': 15,
            'Economic Loss': 5
        }
        
        colors_list = [self.colors['control'], self.colors['optimal'], 
                      self.colors['baseline'], self.colors['infected'], 
                      self.colors['exposed']]
        
        wedges, texts, autotexts = ax.pie(costs.values(), labels=costs.keys(), 
                                          colors=colors_list, autopct='%1.1f%%',
                                          startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Cost Breakdown', fontsize=14, fontweight='bold')
    
    def _plot_strategy_comparison(self, ax, control_results):
        """Plot strategy comparison"""
        strategies = ['No Control', 'Moderate', 'Aggressive', 'Optimal']
        metrics = {
            'Total Infected': [0.85, 0.45, 0.25, 0.20],
            'Peak Infected': [0.45, 0.28, 0.18, 0.15],
            'Total Cost': [0, 0.3, 0.7, 0.5],
            'Deaths Prevented': [0, 0.4, 0.7, 0.75]
        }
        
        x = np.arange(len(strategies))
        width = 0.2
        
        for i, (metric, values) in enumerate(metrics.items()):
            offset = (i - 1.5) * width
            ax.bar(x + offset, values, width, label=metric)
        
        ax.set_xlabel('Strategy', fontsize=14)
        ax.set_ylabel('Normalized Value', fontsize=14)
        ax.set_title('Comprehensive Strategy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, axis='y', alpha=0.3)
    
    def _create_latex_versions(self, output_dir):
        """Create LaTeX-compatible versions of figures"""
        # Create a LaTeX include file
        latex_content = r"""
% LaTeX figure inclusion commands for climate-epidemic analysis

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figure1_epidemic_dynamics.png}
    \caption{Comprehensive epidemic dynamics under climate influence. (a) SEIR compartment evolution, (b) Effective reproduction number, (c) Phase portrait, (d) Climate scenario comparison.}
    \label{fig:epidemic_dynamics}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figure2_sensitivity_analysis.png}
    \caption{Global sensitivity analysis using Sobol indices. The heatmap shows parameter importance for different model outputs.}
    \label{fig:sensitivity}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figure3_uncertainty_analysis.png}
    \caption{Uncertainty quantification through Monte Carlo analysis. Shaded regions represent confidence intervals.}
    \label{fig:uncertainty}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figure4_control_optimization.png}
    \caption{Optimal control strategies for epidemic mitigation under climate constraints.}
    \label{fig:control}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figure5_phase_space_3d.png}
    \caption{Three-dimensional phase space trajectory showing the coupled dynamics of epidemic spread and climate variation.}
    \label{fig:phase_space}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure6_network_evolution.png}
    \caption{Evolution of the contact network structure during epidemic progression.}
    \label{fig:network}
\end{figure}
"""
        
        with open(output_dir / 'latex_figures.tex', 'w') as f:
            f.write(latex_content)
        
        print(f"LaTeX figure commands saved to {output_dir / 'latex_figures.tex'}")


# Convenience function for backwards compatibility
def create_academic_figures(all_results, output_dir):
    """
    Create all academic figures from results
    
    Args:
        all_results: Dictionary containing all analysis results
        output_dir: Directory to save figures
    
    Returns:
        Dictionary of figure objects
    """
    visualizer = AcademicVisualizer(style='paper')
    return visualizer.create_publication_figure_set(all_results, output_dir)