"""
Visualization utilities for climate-epidemic analysis
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set academic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


class Visualizer:
    """Simple visualization class for climate-epidemic analysis"""
    
    def __init__(self):
        self.colors = {
            'S': '#1f77b4',  # Blue
            'E': '#ff7f0e',  # Orange  
            'I': '#d62728',  # Red
            'R': '#2ca02c',  # Green
            'T': '#9467bd',  # Purple
        }
    
    def plot_epidemic_dynamics(self, results, title="Epidemic Dynamics", save_path=None):
        """Plot SEIR compartments over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot compartments
        ax1.plot(results['t'], results['S'], color=self.colors['S'], 
                label='Susceptible', linewidth=2)
        ax1.plot(results['t'], results['E'], color=self.colors['E'], 
                label='Exposed', linewidth=2)
        ax1.plot(results['t'], results['I'], color=self.colors['I'], 
                label='Infected', linewidth=2)
        ax1.plot(results['t'], results['R'], color=self.colors['R'], 
                label='Recovered', linewidth=2)
        
        ax1.set_ylabel('Population Fraction')
        ax1.set_title(title)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot temperature
        ax2.plot(results['t'], results['T'], color=self.colors['T'], linewidth=2)
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Temperature (°C)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
    
    def plot_phase_portrait(self, S, I, title="Phase Portrait", save_path=None):
        """Plot S-I phase portrait"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Main trajectory
        ax.plot(S, I, 'b-', linewidth=2, alpha=0.8)
        
        # Start and end points
        ax.scatter(S[0], I[0], color='green', s=100, marker='o', 
                  label='Start', zorder=5, edgecolors='black')
        ax.scatter(S[-1], I[-1], color='red', s=100, marker='s', 
                  label='End', zorder=5, edgecolors='black')
        
        # Direction arrows
        n_arrows = 5
        indices = np.linspace(0, len(S)-2, n_arrows, dtype=int)
        for idx in indices:
            ax.annotate('', xy=(S[idx+1], I[idx+1]), xytext=(S[idx], I[idx]),
                       arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))
        
        ax.set_xlabel('Susceptible Fraction')
        ax.set_ylabel('Infected Fraction')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
    
    def plot_sensitivity_heatmap(self, sensitivity_matrix, params, outputs, save_path=None):
        """Plot sensitivity analysis heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(sensitivity_matrix, 
                   xticklabels=outputs,
                   yticklabels=params,
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Sensitivity Index'},
                   vmin=0, vmax=1,
                   ax=ax)
        
        ax.set_title('Parameter Sensitivity Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
    
    def plot_control_comparison(self, strategies, metrics, save_path=None):
        """Plot control strategy comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Strategy comparison bars
        x = np.arange(len(strategies))
        width = 0.25
        
        for i, (metric, values) in enumerate(metrics.items()):
            offset = (i - len(metrics)/2 + 0.5) * width
            ax1.bar(x + offset, values, width, label=metric, alpha=0.8)
        
        ax1.set_xlabel('Strategy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies)
        ax1.legend()
        ax1.set_title('Strategy Performance Metrics')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Cost-effectiveness scatter
        if 'Cost' in metrics and 'Effectiveness' in metrics:
            costs = metrics['Cost']
            effectiveness = metrics['Effectiveness']
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
            ax2.scatter(costs, effectiveness, s=200, c=colors, alpha=0.7, 
                       edgecolors='black', linewidth=2)
            
            for i, strategy in enumerate(strategies):
                ax2.annotate(strategy, (costs[i], effectiveness[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax2.set_xlabel('Cost (Million $)')
            ax2.set_ylabel('Effectiveness (%)')
            ax2.set_title('Cost-Effectiveness Analysis')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
    
    def plot_uncertainty_bands(self, t, trajectories, save_path=None):
        """Plot uncertainty quantification with confidence bands"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate percentiles
        p5 = np.percentile(trajectories, 5, axis=0)
        p25 = np.percentile(trajectories, 25, axis=0)
        p50 = np.percentile(trajectories, 50, axis=0)
        p75 = np.percentile(trajectories, 75, axis=0)
        p95 = np.percentile(trajectories, 95, axis=0)
        
        # Plot bands
        ax.fill_between(t, p5, p95, alpha=0.2, color='blue', label='90% CI')
        ax.fill_between(t, p25, p75, alpha=0.4, color='blue', label='50% CI')
        ax.plot(t, p50, 'b-', linewidth=2.5, label='Median')
        
        # Add some sample trajectories
        n_samples = min(20, len(trajectories))
        for i in range(0, len(trajectories), len(trajectories)//n_samples):
            ax.plot(t, trajectories[i], 'gray', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Infected Fraction')
        ax.set_title('Uncertainty Quantification')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
    
    def create_summary_figure(self, all_results, save_path=None):
        """Create a comprehensive summary figure"""
        fig = plt.figure(figsize=(16, 12))
        
        # Grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main dynamics
        ax1 = fig.add_subplot(gs[0, :2])
        for scenario in ['baseline', 'heatwave', 'extreme']:
            if scenario in all_results:
                data = all_results[scenario]
                ax1.plot(data['t'], data['I'], label=scenario.capitalize(), linewidth=2)
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Infected Fraction')
        ax1.set_title('Epidemic Curves Under Different Scenarios')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Phase portrait
        ax2 = fig.add_subplot(gs[0, 2])
        if 'baseline' in all_results:
            data = all_results['baseline']
            ax2.plot(data['S'], data['I'], 'b-', linewidth=2)
            ax2.set_xlabel('S')
            ax2.set_ylabel('I')
            ax2.set_title('Phase Portrait')
            ax2.grid(True, alpha=0.3)
        
        # 3. Temperature profiles
        ax3 = fig.add_subplot(gs[1, :])
        for scenario in ['baseline', 'heatwave', 'extreme']:
            if scenario in all_results:
                data = all_results[scenario]
                ax3.plot(data['t'], data['T'], label=scenario.capitalize(), linewidth=2)
        ax3.set_xlabel('Time (days)')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title('Temperature Profiles')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Calculate summary stats
        summary_text = "Summary Statistics:\n\n"
        for scenario in ['baseline', 'heatwave', 'extreme']:
            if scenario in all_results:
                data = all_results[scenario]
                peak_infected = np.max(data['I'])
                total_infected = np.trapz(data['I'], data['t'])
                summary_text += f"{scenario.capitalize()}:\n"
                summary_text += f"  Peak Infected: {peak_infected:.3f}\n"
                summary_text += f"  Total Attack Rate: {total_infected/365:.3f}\n\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Climate-Epidemic Analysis Summary', fontsize=18)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
