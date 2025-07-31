"""
Visualization utilities for the climate-social network resilience system
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import Rectangle
import pandas as pd

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SystemVisualizer:
    """Visualization tools for system analysis"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        
    def plot_epidemic_dynamics(self, t, y, T_func, title="Epidemic Dynamics"):
        """Plot epidemic compartments over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Compartments
        S, E, I, R, k_avg, C = y
        
        ax1.plot(t, S, label='Susceptible', linewidth=2)
        ax1.plot(t, E, label='Exposed', linewidth=2)
        ax1.plot(t, I, label='Infectious', linewidth=2)
        ax1.plot(t, R, label='Recovered', linewidth=2)
        
        ax1.set_ylabel('Population')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(title)
        
        # Temperature overlay
        T_values = [T_func(time) for time in t]
        ax2_temp = ax2.twinx()
        ax2_temp.plot(t, T_values, 'r--', alpha=0.7, label='Temperature')
        ax2_temp.set_ylabel('Temperature (°C)', color='red')
        ax2_temp.tick_params(axis='y', labelcolor='red')
        
        # Network metrics
        ax2.plot(t, k_avg, 'g-', linewidth=2, label='Avg Degree')
        ax2.plot(t, C * 20, 'b-', linewidth=2, label='Clustering × 20')
        
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Network Metrics')
        ax2.legend(loc='upper left')
        ax2_temp.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/epidemic_dynamics.png', dpi=300)
        return fig
    
    def plot_phase_portrait(self, results_dict, variables=['I', 'k_avg']):
        """Plot phase portrait of system dynamics"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        var1_idx = ['S', 'E', 'I', 'R', 'k_avg', 'C'].index(variables[0])
        var2_idx = ['S', 'E', 'I', 'R', 'k_avg', 'C'].index(variables[1])
        
        for name, result in results_dict.items():
            if 'y' in result:
                y = result['y']
                ax.plot(y[var1_idx], y[var2_idx], label=name, linewidth=2)
                # Mark starting point
                ax.plot(y[var1_idx, 0], y[var2_idx, 0], 'o', markersize=8)
        
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Phase Portrait: {variables[0]} vs {variables[1]}')
        plt.savefig('figures/phase_portrait.png', dpi=300)
        return fig
    
    def plot_bifurcation_diagram(self, bifurcation_results):
        """Plot bifurcation diagram"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(self.figsize[0], 12), sharex=True)
        
        temperatures = bifurcation_results['temperatures']
        equilibria = bifurcation_results['equilibria']
        stability = bifurcation_results['stability']
        R0_values = bifurcation_results['R0_values']
        
        # Extract infection levels and network degrees
        I_eq = [eq[2] for eq in equilibria]
        k_eq = [eq[4] for eq in equilibria]
        
        # Color code by stability
        colors = ['red' if s == 'unstable' else 'blue' if s == 'stable' else 'orange' 
                 for s in stability]
        
        # Infections
        ax1.scatter(temperatures, I_eq, c=colors, alpha=0.7)
        ax1.set_ylabel('Infections at Equilibrium')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Bifurcation Diagram')
        
        # Network degree
        ax2.scatter(temperatures, k_eq, c=colors, alpha=0.7)
        ax2.set_ylabel('Network Degree at Equilibrium')
        ax2.grid(True, alpha=0.3)
        
        # R0
        ax3.plot(temperatures, R0_values, 'k-', linewidth=2)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='R₀ = 1')
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Basic Reproduction Number')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Legend for stability
        stable_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='blue', markersize=8, label='Stable')
        unstable_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor='red', markersize=8, label='Unstable')
        ax1.legend(handles=[stable_patch, unstable_patch], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('figures/bifurcation_diagram.png', dpi=300)
        return fig
    
    def plot_control_strategies(self, strategy_results):
        """Compare different control strategies"""
        n_strategies = len(strategy_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Colors for strategies
        colors = plt.cm.Set3(np.linspace(0, 1, n_strategies))
        
        # Plot 1: Infection trajectories
        for i, (name, result) in enumerate(strategy_results.items()):
            if 'y' in result:
                t = result['t']
                I = result['y'][2]  # Infections
                axes[0].plot(t, I, label=name, color=colors[i], linewidth=2)
        
        axes[0].set_xlabel('Time (days)')
        axes[0].set_ylabel('Infections')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Infection Trajectories')
        
        # Plot 2: Network degree evolution
        for i, (name, result) in enumerate(strategy_results.items()):
            if 'y' in result:
                t = result['t']
                k_avg = result['y'][4]  # Average degree
                axes[1].plot(t, k_avg, label=name, color=colors[i], linewidth=2)
        
        axes[1].set_xlabel('Time (days)')
        axes[1].set_ylabel('Average Network Degree')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Network Evolution')
        
        # Plot 3: Resilience scores
        for i, (name, result) in enumerate(strategy_results.items()):
            if 'resilience_scores' in result:
                t = result['t']
                resilience = result['resilience_scores']
                axes[2].plot(t, resilience, label=name, color=colors[i], linewidth=2)
        
        axes[2].set_xlabel('Time (days)')
        axes[2].set_ylabel('System Resilience')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Resilience Evolution')
        
        # Plot 4: Summary metrics
        metrics = ['total_infections', 'peak_infections', 'final_network_degree']
        strategy_names = list(strategy_results.keys())
        
        data_matrix = []
        for metric in metrics:
            row = []
            for name in strategy_names:
                if metric in strategy_results[name]:
                    row.append(strategy_results[name][metric])
                else:
                    row.append(0)
            data_matrix.append(row)
        
        im = axes[3].imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
        axes[3].set_xticks(range(len(strategy_names)))
        axes[3].set_xticklabels(strategy_names, rotation=45)
        axes[3].set_yticks(range(len(metrics)))
        axes[3].set_yticklabels(metrics)
        axes[3].set_title('Strategy Comparison Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[3])
        cbar.set_label('Normalized Values')
        
        plt.tight_layout()
        plt.savefig('figures/control_strategies.png', dpi=300)
        return fig
    
    def plot_sensitivity_analysis(self, sobol_indices):
        """Visualize sensitivity analysis results"""
        metrics = list(sobol_indices.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if 'first_order' in sobol_indices[metric]:
                params = list(sobol_indices[metric]['first_order'].keys())
                indices = list(sobol_indices[metric]['first_order'].values())
                
                # Sort by magnitude
                sorted_data = sorted(zip(params, indices), key=lambda x: abs(x[1]), reverse=True)
                sorted_params, sorted_indices = zip(*sorted_data)
                
                bars = axes[i].barh(range(len(sorted_params)), sorted_indices)
                axes[i].set_yticks(range(len(sorted_params)))
                axes[i].set_yticklabels(sorted_params)
                axes[i].set_xlabel('First-order Sobol Index')
                axes[i].set_title(f'Sensitivity: {metric}')
                axes[i].grid(True, alpha=0.3)
                
                # Color bars by magnitude
                for j, bar in enumerate(bars):
                    if sorted_indices[j] > 0:
                        bar.set_color('steelblue')
                    else:
                        bar.set_color('coral')
        
        plt.tight_layout()
        plt.savefig('figures/sensitivity_analysis.png', dpi=300)
        return fig
    
    def plot_uncertainty_analysis(self, mc_results, bounds):
        """Plot uncertainty quantification results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Convert results to DataFrame for easier plotting
        df = pd.DataFrame(mc_results)
        
        # Plot 1: Peak infections distribution
        if 'peak_infections' in df.columns:
            axes[0].hist(df['peak_infections'].dropna(), bins=30, alpha=0.7, 
                        color='steelblue', edgecolor='black')
            if 'peak_infections' in bounds:
                axes[0].axvline(bounds['peak_infections']['mean'], color='red', 
                               linestyle='--', label='Mean')
                axes[0].axvline(bounds['peak_infections']['lower'], color='orange', 
                               linestyle=':', label=f"95% CI")
                axes[0].axvline(bounds['peak_infections']['upper'], color='orange', 
                               linestyle=':')
            axes[0].set_xlabel('Peak Infections')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            axes[0].set_title('Peak Infections Distribution')
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Attack rate distribution
        if 'attack_rate' in df.columns:
            axes[1].hist(df['attack_rate'].dropna(), bins=30, alpha=0.7, 
                        color='forestgreen', edgecolor='black')
            if 'attack_rate' in bounds:
                axes[1].axvline(bounds['attack_rate']['mean'], color='red', 
                               linestyle='--', label='Mean')
            axes[1].set_xlabel('Final Attack Rate')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            axes[1].set_title('Attack Rate Distribution')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Resilience vs Network degradation
        if 'min_resilience' in df.columns and 'network_degradation' in df.columns:
            scatter = axes[2].scatter(df['network_degradation'], df['min_resilience'], 
                                     alpha=0.6, c=df['peak_infections'], cmap='viridis')
            axes[2].set_xlabel('Network Degradation')
            axes[2].set_ylabel('Minimum Resilience')
            axes[2].set_title('Resilience vs Network Impact')
            plt.colorbar(scatter, ax=axes[2], label='Peak Infections')
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        im = axes[3].imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        axes[3].set_xticks(range(len(corr_matrix.columns)))
        axes[3].set_yticks(range(len(corr_matrix.columns)))
        axes[3].set_xticklabels(corr_matrix.columns, rotation=45)
        axes[3].set_yticklabels(corr_matrix.columns)
        axes[3].set_title('Output Correlations')
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                axes[3].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                            ha='center', va='center')
        
        plt.colorbar(im, ax=axes[3])
        plt.tight_layout()
        plt.savefig('figures/uncertainty_analysis.png', dpi=300)
        return fig
    
    def plot_network_evolution(self, G_initial, G_final, pos=None):
        """Visualize network structure changes"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if pos is None:
            pos = nx.spring_layout(G_initial, seed=42)
        
        # Initial network
        nx.draw(G_initial, pos, ax=ax1, node_color='lightblue', 
                node_size=50, alpha=0.8, width=0.5)
        ax1.set_title(f'Initial Network\nNodes: {G_initial.number_of_nodes()}, '
                     f'Edges: {G_initial.number_of_edges()}')
        
        # Final network
        nx.draw(G_final, pos, ax=ax2, node_color='lightcoral', 
                node_size=50, alpha=0.8, width=0.5)
        ax2.set_title(f'Final Network\nNodes: {G_final.number_of_nodes()}, '
                     f'Edges: {G_final.number_of_edges()}')
        plt.tight_layout()
        plt.savefig('figures/network_evolution.png', dpi=300)
        return fig
