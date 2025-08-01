"""
Visualization utilities for the climate-social network resilience system
"""
import os
import matplotlib
# Set non-interactive backend for headless environments (GitHub Actions)
matplotlib.use('Agg')
# Disable font caching and use system fonts to avoid emoji font issues
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
# Disable automatic font rebuilding
os.environ['MPLCONFIGDIR'] = '/tmp'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import Rectangle
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
# Ensure tight layout by default
plt.rcParams['figure.autolayout'] = True

class SystemVisualizer:
    """Visualization tools for system analysis"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.save_dpi = 300  # High quality for papers
        
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

    def plot_control_trajectories(self, results_dict, t_span, save_path='figures/control_trajectories.png'):
        """Plot optimal control trajectories over time"""
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        control_names = ['Medical Intervention', 'Social Distancing', 'Climate Mitigation']
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        # Get optimal control values
        if 'optimal' in results_dict and 'control_values' in results_dict['optimal']:
            t = results_dict['optimal']['t']
            control_values = np.array(results_dict['optimal']['control_values']).T
            
            # Plot each control
            for i in range(3):
                ax = fig.add_subplot(gs[i, 0])
                ax.plot(t[::max(1, len(t)//len(control_values[0]))], control_values[i], 
                       color=colors[i], linewidth=2.5, label='Optimal')
                ax.fill_between(t[::max(1, len(t)//len(control_values[0]))], 
                               0, control_values[i], alpha=0.3, color=colors[i])
                ax.set_ylabel(f'{control_names[i]}\nControl Level', fontsize=11)
                ax.set_ylim([0, 1.05])
                ax.grid(True, alpha=0.3)
                if i == 2:
                    ax.set_xlabel('Time (days)', fontsize=11)
                else:
                    ax.set_xticklabels([])
                ax.legend()
        
        # Combined control plot
        ax_combined = fig.add_subplot(gs[:, 1])
        if 'optimal' in results_dict and 'control_values' in results_dict['optimal']:
            t_plot = t[::max(1, len(t)//len(control_values[0]))]
            
            # Stacked area plot
            ax_combined.fill_between(t_plot, 0, control_values[0], 
                                   color=colors[0], alpha=0.7, label=control_names[0])
            ax_combined.fill_between(t_plot, control_values[0], 
                                   control_values[0] + control_values[1], 
                                   color=colors[1], alpha=0.7, label=control_names[1])
            ax_combined.fill_between(t_plot, control_values[0] + control_values[1], 
                                   control_values[0] + control_values[1] + control_values[2], 
                                   color=colors[2], alpha=0.7, label=control_names[2])
            
            ax_combined.set_xlabel('Time (days)', fontsize=12)
            ax_combined.set_ylabel('Combined Control Effort', fontsize=12)
            ax_combined.legend(loc='upper right')
            ax_combined.grid(True, alpha=0.3)
            ax_combined.set_title('Optimal Control Strategy Over Time', fontsize=14, fontweight='bold')
        
        plt.suptitle('Optimal Control Trajectories for Climate-Epidemic System', fontsize=16)
        plt.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
        return fig
    
    def plot_3d_phase_space(self, results_dict, variables=['S', 'I', 'k_avg'], 
                           save_path='figures/3d_phase_space.png'):
        """Plot 3D phase space trajectory"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Variable indices
        var_indices = {var: ['S', 'E', 'I', 'R', 'k_avg', 'C'].index(var) 
                      for var in variables}
        
        # Color map for different strategies
        strategy_colors = {
            'no_control': '#7f8c8d',
            'medical_only': '#e74c3c',
            'social_only': '#3498db',
            'climate_only': '#2ecc71',
            'optimal': '#9b59b6'
        }
        
        for strategy, result in results_dict.items():
            if 'y' in result and strategy in strategy_colors:
                y = result['y']
                
                # Downsample for clarity
                step = max(1, len(y[0]) // 500)
                
                ax.plot(y[var_indices[variables[0]], ::step],
                       y[var_indices[variables[1]], ::step],
                       y[var_indices[variables[2]], ::step],
                       color=strategy_colors[strategy],
                       linewidth=2,
                       label=strategy.replace('_', ' ').title(),
                       alpha=0.8)
                
                # Mark start and end points
                ax.scatter(y[var_indices[variables[0]], 0],
                          y[var_indices[variables[1]], 0],
                          y[var_indices[variables[2]], 0],
                          color=strategy_colors[strategy],
                          s=100, marker='o', edgecolor='black', linewidth=2)
                
                ax.scatter(y[var_indices[variables[0]], -1],
                          y[var_indices[variables[1]], -1],
                          y[var_indices[variables[2]], -1],
                          color=strategy_colors[strategy],
                          s=100, marker='s', edgecolor='black', linewidth=2)
        
        ax.set_xlabel(f'{variables[0]} (Population)', fontsize=12)
        ax.set_ylabel(f'{variables[1]} (Population)', fontsize=12)
        ax.set_zlabel(f'{variables[2]}', fontsize=12)
        ax.set_title('3D Phase Space: System Trajectories', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        plt.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
        return fig
    
    def plot_sensitivity_heatmap(self, sensitivity_results, save_path='figures/sensitivity_heatmap.png'):
        """Plot sensitivity analysis results as heatmap"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Extract sensitivity indices
        params = list(sensitivity_results.keys())
        metrics = ['peak_infections', 'total_infections', 'final_network', 'avg_resilience']
        
        # Create data matrix
        data_matrix = np.zeros((len(params), len(metrics)))
        
        for i, param in enumerate(params):
            if isinstance(sensitivity_results[param], dict):
                data_matrix[i, 0] = sensitivity_results[param].get('peak_infections_sensitivity', 0)
                data_matrix[i, 1] = sensitivity_results[param].get('total_infections_sensitivity', 0)
                data_matrix[i, 2] = sensitivity_results[param].get('network_sensitivity', 0)
                data_matrix[i, 3] = sensitivity_results[param].get('resilience_sensitivity', 0)
        
        # Main heatmap
        ax_main = axes[0, 0]
        im = ax_main.imshow(data_matrix.T, aspect='auto', cmap='RdBu_r', 
                           vmin=-np.max(np.abs(data_matrix)), 
                           vmax=np.max(np.abs(data_matrix)))
        
        ax_main.set_xticks(range(len(params)))
        ax_main.set_xticklabels([p.replace('_', ' ').title() for p in params], rotation=45, ha='right')
        ax_main.set_yticks(range(len(metrics)))
        ax_main.set_yticklabels(['Peak Infections', 'Total Infections', 'Final Network', 'Avg Resilience'])
        ax_main.set_title('Sensitivity Analysis Heatmap', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_main)
        cbar.set_label('Sensitivity Index', rotation=270, labelpad=20)
        
        # Individual metric plots
        for idx, (metric, title) in enumerate(zip(metrics[1:], 
                                                  ['Total Infections', 'Network Structure', 'System Resilience'])):
            ax = axes.flatten()[idx + 1]
            values = data_matrix[:, idx + 1]
            sorted_indices = np.argsort(np.abs(values))[::-1]
            
            colors = ['#e74c3c' if v > 0 else '#3498db' for v in values[sorted_indices]]
            bars = ax.barh(range(len(params)), values[sorted_indices], color=colors)
            
            ax.set_yticks(range(len(params)))
            ax.set_yticklabels([params[i].replace('_', ' ').title() for i in sorted_indices])
            ax.set_xlabel('Sensitivity Index')
            ax.set_title(f'Sensitivity: {title}', fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
            ax.axvline(x=0, color='black', linewidth=0.5)
        
        plt.suptitle('Parameter Sensitivity Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
        return fig
    
    def plot_strategy_comparison_dashboard(self, results_dict, save_path='figures/strategy_dashboard.png'):
        """Create comprehensive dashboard comparing all strategies"""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        strategies = list(results_dict.keys())
        strategy_colors = {
            'no_control': '#7f8c8d',
            'medical_only': '#e74c3c',
            'social_only': '#3498db',
            'climate_only': '#2ecc71',
            'optimal': '#9b59b6'
        }
        
        # 1. Infection dynamics comparison
        ax1 = fig.add_subplot(gs[0, :2])
        for strategy in strategies:
            if 'y' in results_dict[strategy]:
                t = results_dict[strategy]['t']
                I = results_dict[strategy]['y'][2]  # Infectious
                ax1.plot(t, I, label=strategy.replace('_', ' ').title(), 
                        color=strategy_colors.get(strategy, 'gray'), linewidth=2)
        
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Infectious Population')
        ax1.set_title('Infection Dynamics Comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Network evolution
        ax2 = fig.add_subplot(gs[1, :2])
        for strategy in strategies:
            if 'y' in results_dict[strategy]:
                t = results_dict[strategy]['t']
                k_avg = results_dict[strategy]['y'][4]
                ax2.plot(t, k_avg, label=strategy.replace('_', ' ').title(),
                        color=strategy_colors.get(strategy, 'gray'), linewidth=2)
        
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Average Network Degree')
        ax2.set_title('Network Structure Evolution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance metrics comparison
        ax3 = fig.add_subplot(gs[2, :])
        metrics = ['peak_infections', 'total_infections', 'final_network_degree']
        metric_labels = ['Peak Infections', 'Total Infections', 'Final Network Degree']
        
        x = np.arange(len(strategies))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = []
            for strategy in strategies:
                if metric in results_dict[strategy]:
                    value = results_dict[strategy][metric]
                    # Normalize values for comparison
                    if metric == 'total_infections':
                        value = value / 1000  # Scale down
                    values.append(value)
                else:
                    values.append(0)
            
            ax3.bar(x + i * width, values, width, label=metric_labels[i])
        
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Normalized Value')
        ax3.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([s.replace('_', ' ').title() for s in strategies], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Resilience scores
        ax4 = fig.add_subplot(gs[0, 2])
        resilience_data = []
        labels = []
        
        for strategy in strategies:
            if 'resilience_scores' in results_dict[strategy]:
                scores = results_dict[strategy]['resilience_scores']
                if scores:
                    resilience_data.append(scores)
                    labels.append(strategy.replace('_', ' ').title())
        
        if resilience_data:
            bp = ax4.boxplot(resilience_data, labels=labels, patch_artist=True)
            for patch, strategy in zip(bp['boxes'], strategies):
                patch.set_facecolor(strategy_colors.get(strategy, 'gray'))
            
            ax4.set_ylabel('Resilience Score')
            ax4.set_title('System Resilience Distribution', fontsize=12, fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Cost-effectiveness radar chart
        ax5 = fig.add_subplot(gs[1:, 2], projection='polar')
        
        # Define metrics for radar chart
        categories = ['Health\nOutcome', 'Network\nPreservation', 'Cost\nEfficiency', 
                     'Resilience', 'Implementation\nEase']
        
        # Calculate scores for each strategy (normalized 0-1)
        for strategy in ['no_control', 'optimal']:
            if strategy in results_dict and 'y' in results_dict[strategy]:
                result = results_dict[strategy]
                
                # Calculate normalized scores
                health_score = 1 - (result.get('peak_infections', 1) / 
                                  max([r.get('peak_infections', 1) for r in results_dict.values()]))
                network_score = result.get('final_network_degree', 0) / 20  # Normalize by baseline
                cost_score = 1 if strategy == 'no_control' else 0.7  # Placeholder
                resilience_score = np.mean(result.get('resilience_scores', [0.5]))
                implementation_score = 1 if strategy == 'no_control' else 0.6
                
                scores = [health_score, network_score, cost_score, 
                         resilience_score, implementation_score]
                
                # Plot
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                scores += scores[:1]  # Complete the circle
                angles += angles[:1]
                
                ax5.plot(angles, scores, 'o-', linewidth=2, 
                        label=strategy.replace('_', ' ').title(),
                        color=strategy_colors.get(strategy, 'gray'))
                ax5.fill(angles, scores, alpha=0.25, 
                        color=strategy_colors.get(strategy, 'gray'))
        
        ax5.set_theta_offset(np.pi / 2)
        ax5.set_theta_direction(-1)
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories)
        ax5.set_ylim(0, 1)
        ax5.set_title('Multi-Criteria Performance', fontsize=12, fontweight='bold', pad=20)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax5.grid(True)
        
        plt.suptitle('Comprehensive Strategy Comparison Dashboard', fontsize=18, fontweight='bold')
        plt.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
        return fig
    
    def plot_uncertainty_bands(self, monte_carlo_results, save_path='figures/uncertainty_bands.png'):
        """Plot results with uncertainty bands from Monte Carlo analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if 'trajectories' in monte_carlo_results:
            trajectories = monte_carlo_results['trajectories']
            t = monte_carlo_results.get('time', np.arange(trajectories.shape[2]))
            
            # Variables to plot
            var_names = ['Susceptible', 'Infected', 'Network Degree', 'Clustering']
            var_indices = [0, 2, 4, 5]
            
            for idx, (ax, var_idx, var_name) in enumerate(zip(axes.flatten(), var_indices, var_names)):
                # Calculate percentiles
                data = trajectories[:, var_idx, :]
                median = np.median(data, axis=0)
                p25 = np.percentile(data, 25, axis=0)
                p75 = np.percentile(data, 75, axis=0)
                p5 = np.percentile(data, 5, axis=0)
                p95 = np.percentile(data, 95, axis=0)
                
                # Plot
                ax.plot(t, median, 'b-', linewidth=2, label='Median')
                ax.fill_between(t, p25, p75, alpha=0.3, color='blue', label='50% CI')
                ax.fill_between(t, p5, p95, alpha=0.1, color='blue', label='90% CI')
                
                # Add some individual trajectories
                for i in range(min(5, len(data))):
                    ax.plot(t, data[i], 'gray', alpha=0.2, linewidth=0.5)
                
                ax.set_xlabel('Time (days)')
                ax.set_ylabel(var_name)
                ax.set_title(f'{var_name} with Uncertainty', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                if idx == 0:
                    ax.legend()
        
        plt.suptitle('Monte Carlo Uncertainty Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
        return fig
    
    def create_publication_figure_set(self, all_results, save_dir='figures/publication'):
        """Create a complete set of publication-ready figures"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        figures = {}
        
        # 1. Main results figure
        if 'control_comparison' in all_results:
            fig = self.plot_strategy_comparison_dashboard(
                all_results['control_comparison'],
                save_path=f'{save_dir}/main_results.png'
            )
            figures['main_results'] = fig
            plt.close(fig)
        
        # 2. Control trajectories
        if 'control_comparison' in all_results:
            fig = self.plot_control_trajectories(
                all_results['control_comparison'],
                t_span=[0, 365],
                save_path=f'{save_dir}/control_trajectories.png'
            )
            figures['control_trajectories'] = fig
            plt.close(fig)
        
        # 3. 3D phase space
        if 'control_comparison' in all_results:
            fig = self.plot_3d_phase_space(
                all_results['control_comparison'],
                save_path=f'{save_dir}/phase_space_3d.png'
            )
            figures['phase_space_3d'] = fig
            plt.close(fig)
        
        # 4. Sensitivity analysis
        if 'sensitivity' in all_results:
            fig = self.plot_sensitivity_heatmap(
                all_results['sensitivity'],
                save_path=f'{save_dir}/sensitivity_analysis.png'
            )
            figures['sensitivity'] = fig
            plt.close(fig)
        
        # 5. Uncertainty analysis
        if 'uncertainty' in all_results:
            fig = self.plot_uncertainty_bands(
                all_results['uncertainty'],
                save_path=f'{save_dir}/uncertainty_analysis.png'
            )
            figures['uncertainty'] = fig
            plt.close(fig)
        
        # 6. Network visualization
        if 'network_snapshots' in all_results:
            fig = self.plot_network_evolution(
                all_results['network_snapshots'],
                save_path=f'{save_dir}/network_evolution.png'
            )
            figures['network_evolution'] = fig
            plt.close(fig)
        
        return figures
    
    def compare_scenarios(self, scenario_data):
        """Compare different scenarios side by side"""
        n_scenarios = len(scenario_data)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        colors = ['blue', 'orange', 'red']
        
        for idx, (scenario_name, data) in enumerate(scenario_data.items()):
            y = data['y']
            params = data['params']
            
            # Plot on all subplots
            # 1. Infections over time
            ax = axes[0]
            ax.plot(y[1, :], label=scenario_name.capitalize(), 
                   color=colors[idx % len(colors)], linewidth=2, alpha=0.8)
            ax.set_ylabel('Infected Population')
            ax.set_xlabel('Time (days)')
            ax.set_title('Infection Dynamics Comparison')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 2. Network degree over time
            ax = axes[1]
            ax.plot(y[4, :], label=scenario_name.capitalize(), 
                   color=colors[idx % len(colors)], linewidth=2, alpha=0.8)
            ax.set_ylabel('Average Network Degree')
            ax.set_xlabel('Time (days)')
            ax.set_title('Network Evolution Comparison')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 3. Peak values comparison (bar chart)
            if idx == 0:
                peak_infections = []
                peak_degrees = []
                scenario_names = []
            
            peak_infections.append(np.max(y[1, :]))
            peak_degrees.append(np.max(y[4, :]))
            scenario_names.append(scenario_name.capitalize())
        
        # 3. Peak infections bar chart
        ax = axes[2]
        x_pos = np.arange(len(scenario_names))
        ax.bar(x_pos, peak_infections, color=colors[:len(scenario_names)], alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenario_names)
        ax.set_ylabel('Peak Infections')
        ax.set_title('Maximum Infection Levels')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Final state comparison
        ax = axes[3]
        width = 0.25
        metrics = ['Susceptible', 'Infected', 'Recovered']
        
        for idx, (scenario_name, data) in enumerate(scenario_data.items()):
            y = data['y']
            final_state = [y[0, -1], y[1, -1], y[2, -1]]
            x_pos = np.arange(len(metrics)) + idx * width
            ax.bar(x_pos, final_state, width, label=scenario_name.capitalize(),
                  color=colors[idx % len(colors)], alpha=0.7)
        
        ax.set_xlabel('Compartment')
        ax.set_ylabel('Final Population')
        ax.set_title('Final State Comparison')
        ax.set_xticks(np.arange(len(metrics)) + width)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def generate_latex_figure_code(self, figure_names, save_path='figures/latex_figures.tex'):
        """Generate LaTeX code for including figures in paper"""
        latex_code = r"""% LaTeX code for including figures in your paper
\documentclass{article}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{float}

\begin{document}

% Main results figure
\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/publication/main_results.png}
    \caption{Comprehensive comparison of control strategies for the coupled climate-epidemic system. 
             The dashboard shows (a) infection dynamics, (b) network evolution, (c) performance metrics, 
             (d) resilience distribution, and (e) multi-criteria evaluation.}
    \label{fig:main_results}
\end{figure}

% Control trajectories
\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/publication/control_trajectories.png}
    \caption{Optimal control trajectories over time. The left panels show individual control 
             components (medical intervention, social distancing, climate mitigation), while the 
             right panel displays the combined control effort.}
    \label{fig:control_trajectories}
\end{figure}

% 3D Phase space
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/publication/phase_space_3d.png}
    \caption{Three-dimensional phase space trajectories comparing different control strategies. 
             Circles indicate initial conditions and squares show final states.}
    \label{fig:phase_space}
\end{figure}

% Sensitivity and uncertainty
\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/publication/sensitivity_analysis.png}
        \caption{Parameter sensitivity analysis}
        \label{fig:sensitivity}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/publication/uncertainty_analysis.png}
        \caption{Uncertainty quantification}
        \label{fig:uncertainty}
    \end{subfigure}
    \caption{(a) Sensitivity analysis showing the impact of parameter variations on key system metrics. 
             (b) Monte Carlo uncertainty analysis with confidence intervals.}
    \label{fig:sensitivity_uncertainty}
\end{figure}

\end{document}
"""
        
        with open(save_path, 'w') as f:
            f.write(latex_code)
        
        return latex_code
