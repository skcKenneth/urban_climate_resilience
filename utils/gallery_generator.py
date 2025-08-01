"""
Visualization gallery generator for climate-epidemic analysis results
"""
import os
import json
from pathlib import Path
from datetime import datetime
import shutil

class GalleryGenerator:
    """Generate HTML galleries for analysis visualizations"""
    
    def __init__(self, results_dir='combined_results', figures_dir='figures'):
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        
    def generate_gallery(self, date_str=None):
        """Generate complete visualization gallery"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        gallery_dir = self.results_dir / date_str
        gallery_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all figures to gallery directory
        figures_dest = gallery_dir / 'figures'
        if self.figures_dir.exists():
            shutil.copytree(self.figures_dir, figures_dest, dirs_exist_ok=True)
        
        # Generate main gallery HTML
        html_content = self._generate_main_gallery(date_str)
        with open(gallery_dir / 'index.html', 'w') as f:
            f.write(html_content)
        
        # Generate category-specific galleries
        categories = {
            'control': 'Control Strategy Analysis',
            'sensitivity': 'Sensitivity Analysis',
            'uncertainty': 'Uncertainty Quantification',
            'network': 'Network Dynamics',
            'epidemic': 'Epidemic Dynamics'
        }
        
        for category, title in categories.items():
            category_html = self._generate_category_gallery(category, title, date_str)
            with open(gallery_dir / f'{category}_gallery.html', 'w') as f:
                f.write(category_html)
        
        # Generate interactive dashboard
        dashboard_html = self._generate_interactive_dashboard(date_str)
        with open(gallery_dir / 'dashboard.html', 'w') as f:
            f.write(dashboard_html)
        
        print(f"✅ Gallery generated at: {gallery_dir}")
        return gallery_dir
    
    def _generate_main_gallery(self, date_str):
        """Generate main gallery HTML"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Climate-Epidemic Analysis Gallery - {date_str}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Climate-Epidemic Analysis Results</h1>
            <p class="subtitle">Generated on {date_str}</p>
        </header>
        
        <nav>
            <ul>
                <li><a href="#overview">Overview</a></li>
                <li><a href="#control">Control Strategies</a></li>
                <li><a href="#sensitivity">Sensitivity Analysis</a></li>
                <li><a href="#uncertainty">Uncertainty</a></li>
                <li><a href="#network">Network Dynamics</a></li>
                <li><a href="dashboard.html">Interactive Dashboard</a></li>
            </ul>
        </nav>
        
        <section id="overview">
            <h2>Analysis Overview</h2>
            <div class="gallery-grid">
                <div class="gallery-item featured">
                    <h3>Strategy Comparison Dashboard</h3>
                    <img src="figures/strategy_dashboard.png" alt="Strategy Dashboard">
                    <p>Comprehensive comparison of all control strategies showing infection dynamics, 
                       network evolution, performance metrics, and multi-criteria evaluation.</p>
                </div>
                
                <div class="gallery-item">
                    <h3>Optimal Control Trajectories</h3>
                    <img src="figures/control_trajectories.png" alt="Control Trajectories">
                    <p>Time evolution of optimal control inputs for medical intervention, 
                       social distancing, and climate mitigation.</p>
                </div>
                
                <div class="gallery-item">
                    <h3>3D Phase Space</h3>
                    <img src="figures/phase_space_3d.png" alt="3D Phase Space">
                    <p>Three-dimensional phase space trajectories comparing different control strategies.</p>
                </div>
            </div>
        </section>
        
        <section id="control">
            <h2>Control Strategy Analysis</h2>
            <div class="gallery-grid">
                <div class="gallery-item">
                    <h3>Phase Portrait: Infections vs Network</h3>
                    <img src="figures/phase_portrait_1.png" alt="Phase Portrait 1">
                    <p>Relationship between infection levels and network connectivity.</p>
                </div>
                
                <div class="gallery-item">
                    <h3>Phase Portrait: Susceptible vs Recovered</h3>
                    <img src="figures/phase_portrait_2.png" alt="Phase Portrait 2">
                    <p>Population dynamics in the S-R phase space.</p>
                </div>
                
                <div class="gallery-item">
                    <h3>Phase Portrait: Infections vs Clustering</h3>
                    <img src="figures/phase_portrait_3.png" alt="Phase Portrait 3">
                    <p>Impact of network clustering on infection spread.</p>
                </div>
            </div>
            <a href="control_gallery.html" class="view-more">View All Control Analysis Results →</a>
        </section>
        
        <section id="sensitivity">
            <h2>Sensitivity Analysis</h2>
            <div class="gallery-grid">
                <div class="gallery-item">
                    <h3>Parameter Sensitivity Heatmap</h3>
                    <img src="figures/sensitivity_heatmap.png" alt="Sensitivity Heatmap">
                    <p>Comprehensive sensitivity analysis showing the impact of parameter variations.</p>
                </div>
                
                <div class="gallery-item">
                    <h3>Tornado Plot</h3>
                    <img src="figures/tornado_plot.png" alt="Tornado Plot">
                    <p>Ranked parameter sensitivities for key system metrics.</p>
                </div>
            </div>
            <a href="sensitivity_gallery.html" class="view-more">View All Sensitivity Results →</a>
        </section>
        
        <section id="uncertainty">
            <h2>Uncertainty Quantification</h2>
            <div class="gallery-grid">
                <div class="gallery-item">
                    <h3>Monte Carlo Uncertainty Bands</h3>
                    <img src="figures/uncertainty_bands.png" alt="Uncertainty Bands">
                    <p>Confidence intervals from Monte Carlo uncertainty analysis.</p>
                </div>
                
                <div class="gallery-item">
                    <h3>Probability Distributions</h3>
                    <img src="figures/mc_distributions.png" alt="MC Distributions">
                    <p>Distribution of key outcomes across Monte Carlo simulations.</p>
                </div>
            </div>
            <a href="uncertainty_gallery.html" class="view-more">View All Uncertainty Results →</a>
        </section>
        
        <section id="network">
            <h2>Network Dynamics</h2>
            <div class="gallery-grid">
                <div class="gallery-item">
                    <h3>Network Evolution</h3>
                    <img src="figures/network_evolution.png" alt="Network Evolution">
                    <p>Time evolution of network structure under different scenarios.</p>
                </div>
                
                <div class="gallery-item">
                    <h3>Epidemic Dynamics</h3>
                    <img src="figures/epidemic_dynamics.png" alt="Epidemic Dynamics">
                    <p>SEIR compartment dynamics with climate and network effects.</p>
                </div>
            </div>
            <a href="network_gallery.html" class="view-more">View All Network Results →</a>
        </section>
        
        <footer>
            <p>Generated by Climate-Epidemic Analysis System v1.0</p>
            <p><a href="dashboard.html">Open Interactive Dashboard</a> | 
               <a href="../README.md">View Documentation</a></p>
        </footer>
    </div>
</body>
</html>"""
    
    def _generate_category_gallery(self, category, title, date_str):
        """Generate category-specific gallery"""
        # Find all figures matching the category
        figures = []
        if (self.results_dir / date_str / 'figures').exists():
            for fig in (self.results_dir / date_str / 'figures').glob('*.png'):
                if category in fig.stem.lower():
                    figures.append(fig.name)
        
        figure_items = '\n'.join([
            f"""<div class="gallery-item">
                <h3>{self._format_figure_name(fig)}</h3>
                <img src="figures/{fig}" alt="{fig}">
                <p>{self._get_figure_description(fig)}</p>
            </div>""" for fig in figures
        ])
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Climate-Epidemic Analysis</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">Generated on {date_str}</p>
            <nav>
                <a href="index.html">← Back to Main Gallery</a>
            </nav>
        </header>
        
        <div class="gallery-grid">
            {figure_items}
        </div>
        
        <footer>
            <p><a href="index.html">← Back to Main Gallery</a></p>
        </footer>
    </div>
</body>
</html>"""
    
    def _generate_interactive_dashboard(self, date_str):
        """Generate interactive dashboard with JavaScript"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Dashboard - Climate-Epidemic Analysis</title>
    <style>
        {self._get_css_styles()}
        .dashboard-controls {{
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .control-group {{
            margin-bottom: 15px;
        }}
        .control-group label {{
            display: inline-block;
            width: 150px;
            font-weight: bold;
        }}
        .image-comparison {{
            position: relative;
            overflow: hidden;
            margin: 20px 0;
        }}
        .comparison-slider {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 50%;
            overflow: hidden;
            border-right: 2px solid #fff;
        }}
        input[type="range"] {{
            width: 300px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Interactive Analysis Dashboard</h1>
            <p class="subtitle">Climate-Epidemic Analysis - {date_str}</p>
            <nav>
                <a href="index.html">← Back to Gallery</a>
            </nav>
        </header>
        
        <section class="dashboard-controls">
            <h2>Interactive Controls</h2>
            
            <div class="control-group">
                <label>Analysis Type:</label>
                <select id="analysisType" onchange="updateDisplay()">
                    <option value="control">Control Strategies</option>
                    <option value="sensitivity">Sensitivity Analysis</option>
                    <option value="uncertainty">Uncertainty Quantification</option>
                    <option value="network">Network Dynamics</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Visualization:</label>
                <select id="vizType" onchange="updateDisplay()">
                    <option value="main">Main Results</option>
                    <option value="phase">Phase Portraits</option>
                    <option value="3d">3D Visualizations</option>
                    <option value="heatmap">Heatmaps</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Comparison Mode:</label>
                <input type="checkbox" id="comparisonMode" onchange="toggleComparison()">
                <span>Enable side-by-side comparison</span>
            </div>
        </section>
        
        <section id="displayArea">
            <div id="mainDisplay" class="gallery-item featured">
                <h3 id="displayTitle">Strategy Comparison Dashboard</h3>
                <img id="displayImage" src="figures/strategy_dashboard.png" alt="Display">
                <p id="displayDescription">Select options above to explore different visualizations.</p>
            </div>
            
            <div id="comparisonDisplay" style="display: none;">
                <div class="gallery-grid">
                    <div class="gallery-item">
                        <h3>Image 1</h3>
                        <img id="compareImage1" src="figures/strategy_dashboard.png" alt="Compare 1">
                    </div>
                    <div class="gallery-item">
                        <h3>Image 2</h3>
                        <img id="compareImage2" src="figures/control_trajectories.png" alt="Compare 2">
                    </div>
                </div>
            </div>
        </section>
        
        <section id="analysisMetrics">
            <h2>Key Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Total Infections</h4>
                    <p class="metric-value" id="metricInfections">-</p>
                </div>
                <div class="metric-card">
                    <h4>Peak Infections</h4>
                    <p class="metric-value" id="metricPeak">-</p>
                </div>
                <div class="metric-card">
                    <h4>Network Preservation</h4>
                    <p class="metric-value" id="metricNetwork">-</p>
                </div>
                <div class="metric-card">
                    <h4>System Resilience</h4>
                    <p class="metric-value" id="metricResilience">-</p>
                </div>
            </div>
        </section>
        
        <footer>
            <p>Interactive Dashboard - Climate-Epidemic Analysis System</p>
        </footer>
    </div>
    
    <script>
        // Image mapping
        const imageMap = {{
            control: {{
                main: 'figures/strategy_dashboard.png',
                phase: 'figures/phase_portrait_1.png',
                '3d': 'figures/phase_space_3d.png',
                heatmap: 'figures/control_trajectories.png'
            }},
            sensitivity: {{
                main: 'figures/sensitivity_heatmap.png',
                phase: 'figures/tornado_plot.png',
                '3d': 'figures/sensitivity_3d.png',
                heatmap: 'figures/sensitivity_heatmap.png'
            }},
            uncertainty: {{
                main: 'figures/uncertainty_bands.png',
                phase: 'figures/mc_distributions.png',
                '3d': 'figures/uncertainty_3d.png',
                heatmap: 'figures/uncertainty_heatmap.png'
            }},
            network: {{
                main: 'figures/network_evolution.png',
                phase: 'figures/epidemic_dynamics.png',
                '3d': 'figures/network_3d.png',
                heatmap: 'figures/network_heatmap.png'
            }}
        }};
        
        // Mock metrics data
        const metricsData = {{
            control: {{
                infections: '12,450',
                peak: '850',
                network: '85%',
                resilience: '0.72'
            }},
            sensitivity: {{
                infections: 'β₀: 0.35',
                peak: 'γ: 0.28',
                network: 'k₀: 0.22',
                resilience: 'α: 0.18'
            }},
            uncertainty: {{
                infections: '12,450 ± 1,200',
                peak: '850 ± 120',
                network: '85% ± 8%',
                resilience: '0.72 ± 0.08'
            }},
            network: {{
                infections: '15,200',
                peak: '1,050',
                network: '65%',
                resilience: '0.58'
            }}
        }};
        
        function updateDisplay() {{
            const analysisType = document.getElementById('analysisType').value;
            const vizType = document.getElementById('vizType').value;
            const imagePath = imageMap[analysisType][vizType] || imageMap[analysisType].main;
            
            document.getElementById('displayImage').src = imagePath;
            document.getElementById('displayTitle').textContent = 
                analysisType.charAt(0).toUpperCase() + analysisType.slice(1) + ' - ' + 
                vizType.charAt(0).toUpperCase() + vizType.slice(1);
            
            // Update metrics
            const metrics = metricsData[analysisType];
            document.getElementById('metricInfections').textContent = metrics.infections;
            document.getElementById('metricPeak').textContent = metrics.peak;
            document.getElementById('metricNetwork').textContent = metrics.network;
            document.getElementById('metricResilience').textContent = metrics.resilience;
        }}
        
        function toggleComparison() {{
            const comparisonMode = document.getElementById('comparisonMode').checked;
            document.getElementById('mainDisplay').style.display = comparisonMode ? 'none' : 'block';
            document.getElementById('comparisonDisplay').style.display = comparisonMode ? 'block' : 'none';
        }}
        
        // Initialize
        updateDisplay();
    </script>
</body>
</html>"""
    
    def _get_css_styles(self):
        """Get CSS styles for galleries"""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        h1 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .subtitle {
            color: #7f8c8d;
            margin: 0;
        }
        nav {
            margin-top: 20px;
        }
        nav ul {
            list-style: none;
            padding: 0;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        nav a {
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
        }
        nav a:hover {
            text-decoration: underline;
        }
        section {
            background: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 20px;
        }
        .gallery-item {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .gallery-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .gallery-item.featured {
            grid-column: span 2;
        }
        .gallery-item h3 {
            margin: 0;
            padding: 15px;
            background: #f8f9fa;
            color: #2c3e50;
            font-size: 1.1em;
        }
        .gallery-item img {
            width: 100%;
            height: auto;
            display: block;
        }
        .gallery-item p {
            padding: 15px;
            margin: 0;
            color: #666;
            font-size: 0.95em;
        }
        .view-more {
            display: inline-block;
            margin-top: 10px;
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
        }
        .view-more:hover {
            text-decoration: underline;
        }
        footer {
            text-align: center;
            padding: 30px;
            color: #7f8c8d;
        }
        footer a {
            color: #3498db;
            text-decoration: none;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-card h4 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin: 0;
        }
        """
    
    def _format_figure_name(self, filename):
        """Format figure filename into readable title"""
        name = filename.replace('.png', '').replace('_', ' ')
        return ' '.join(word.capitalize() for word in name.split())
    
    def _get_figure_description(self, filename):
        """Get description for a figure based on filename"""
        descriptions = {
            'strategy_dashboard': 'Comprehensive comparison of all control strategies',
            'control_trajectories': 'Optimal control inputs over time',
            'phase_space_3d': '3D visualization of system trajectories',
            'sensitivity_heatmap': 'Parameter sensitivity analysis results',
            'uncertainty_bands': 'Monte Carlo uncertainty quantification',
            'network_evolution': 'Evolution of network structure',
            'epidemic_dynamics': 'SEIR compartment dynamics',
            'phase_portrait': 'Phase space analysis of system dynamics',
            'tornado_plot': 'Ranked parameter sensitivities',
            'mc_distributions': 'Monte Carlo simulation distributions'
        }
        
        for key, desc in descriptions.items():
            if key in filename.lower():
                return desc
        
        return 'Analysis visualization result'


if __name__ == "__main__":
    # Example usage
    generator = GalleryGenerator()
    gallery_path = generator.generate_gallery()
    print(f"Gallery created at: {gallery_path}")