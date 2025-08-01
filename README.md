# Urban Climate Resilience Analysis

A mathematical modeling system for analyzing urban climate-social network resilience.

## Quick Start

### Local Installation
```bash
pip install -r requirements.txt
```

### Run Quick Test
```bash
python main.py --quick-mode
```

### Run Full Analysis for Paper
```bash
python run_full_analysis.py
```

## GitHub Actions - Automated Analysis

The project includes automated CI/CD that runs your full analysis on GitHub's servers.

### How to Run Full Analysis on GitHub:

1. **Go to your GitHub repository**
2. **Click on "Actions" tab**
3. **Select "Climate Analysis" workflow**
4. **Click "Run workflow"**
5. **Choose analysis type:**
   - `full` - Complete analysis for paper (all scenarios, 365 days)
   - `quick` - Quick test (30 days)
   - `baseline` - Normal climate only
   - `heatwave` - Heatwave scenario only
   - `extreme` - Extreme climate only
6. **Click "Run workflow" button**

### What Gets Generated for Your Paper:

The full analysis generates all figures needed for your research paper:

- **Simulation Results** (for each scenario):
  - `baseline_simulation.png` - Full 365-day simulation
  - `heatwave_simulation.png` - Heatwave conditions
  - `extreme_simulation.png` - Extreme climate conditions
  - `*_epidemic.png` - Epidemic dynamics plots

- **Analysis Figures**:
  - `bifurcation_diagram.png` - Temperature bifurcation analysis
  - `stability_analysis.png` - System stability analysis
  - `phase_portrait_baseline.png` - Phase space dynamics
  - `morris_sensitivity.png` - Morris sensitivity analysis
  - `sobol_sensitivity.png` - Sobol sensitivity heatmap
  - `scenario_comparison.png` - Comparison across all scenarios
  - `optimal_control.png` - Optimal control strategies

- **Data Files**:
  - `baseline_data.npz` - Baseline scenario data
  - `heatwave_data.npz` - Heatwave scenario data
  - `extreme_data.npz` - Extreme scenario data
  - `analysis_summary.txt` - Summary statistics

### Download Results:

1. Wait for workflow to complete (takes ~30-60 minutes for full analysis)
2. Click on the completed workflow run
3. Scroll down to "Artifacts"
4. Download `paper-analysis-results-{number}.zip`
5. Extract to get all figures and data for your paper

### Workflow Status
[![Climate Analysis](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/climate_analysis.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/climate_analysis.yml)

## Project Structure
```
├── models/              # Mathematical models
├── analysis/            # Analysis modules  
├── utils/               # Utility functions
├── results/             # Output directory
├── main.py              # Quick test entry point
└── run_full_analysis.py # Full paper analysis
```

