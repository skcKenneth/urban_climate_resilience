# Urban Climate Resilience Analysis

A comprehensive mathematical modeling framework for analyzing coupled climate-epidemic-network dynamics in urban environments.

## Overview

This project implements sophisticated mathematical models to study:
- Climate-forced epidemiological dynamics (modified SEIR model)
- Dynamic social network evolution under climate stress
- Coupled system behavior with feedback loops
- Optimal control strategies for resource allocation
- Stability and bifurcation analysis
- Comprehensive sensitivity analysis

## Installation

### Local Setup
```bash
git clone https://github.com/yourusername/urban_climate_resilience.git
cd urban_climate_resilience
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- NumPy, SciPy, Matplotlib
- NetworkX, Pandas, Seaborn
- scikit-learn, joblib, psutil

## Running Analyses

### Quick Analysis (Testing)
```bash
# Run baseline scenario (90 days, 5000 population)
python main.py --quick-mode

# Run specific scenario
python main.py --analysis-type heatwave --quick-mode
```

### Full Paper Analysis (Local)
```bash
# Run complete analysis for paper (365 days, 10000 population)
python run_full_analysis.py

# Results will be in paper_results/ directory
```

## GitHub Actions - Automated Analysis

### Running Quick Analysis (Default)
1. Go to **Actions** tab in GitHub
2. Select **Climate Analysis** workflow
3. Click **Run workflow**
4. Keep default settings:
   - Analysis mode: `quick`
   - Scenario type: `baseline`
5. Click **Run workflow** button

### Running Full Paper Analysis
1. Go to **Actions** tab
2. Select **Climate Analysis** workflow  
3. Click **Run workflow**
4. Change settings:
   - Analysis mode: **`paper`** (IMPORTANT!)
5. Click **Run workflow** button
6. Wait ~30-60 minutes for completion

### What Gets Generated

#### Quick Mode:
- `{scenario}_epidemic.png` - Epidemic dynamics visualization
- `{scenario}_data.npz` - Simulation data
- `{scenario}_summary.txt` - Summary statistics

#### Paper Mode (Full Analysis):
- **Epidemic Dynamics**: 
  - `baseline_epidemic_dynamics.png`
  - `heatwave_epidemic_dynamics.png`
  - `extreme_epidemic_dynamics.png`
- **Stability Analysis**:
  - `stability_analysis.png` - Eigenvalue analysis
  - `bifurcation_diagram.png` - Temperature bifurcation
- **Sensitivity Analysis**:
  - `morris_sensitivity.png` - Morris screening results
  - `sobol_sensitivity.png` - Sobol indices heatmap
- **Comparison & Phase Portraits**:
  - `scenario_comparison.png` - All scenarios comparison
  - `phase_portrait_I_k.png` - Infected vs connectivity
  - `phase_portrait_I_C.png` - Infected vs clustering
- **Optimal Control**:
  - `optimal_control_strategies.png` - Control inputs over time
  - `controlled_vs_uncontrolled.png` - Impact comparison
- **Data & Reports**:
  - `{scenario}_results.npz` - Full simulation data
  - `paper_analysis_summary.txt` - Complete summary

### Downloading Results

1. After workflow completes, click on the run
2. Scroll to **Artifacts** section
3. Download `analysis-results-{number}.zip`
4. Extract to get all figures and data

## Project Structure

```
urban_climate_resilience/
├── models/                 # Mathematical models
│   ├── coupled_system.py   # Main coupled dynamics
│   ├── epidemiological_model.py  # SEIR with climate
│   ├── network_model.py    # Dynamic networks
│   └── optimal_control.py  # Control strategies
├── analysis/              # Analysis modules
│   ├── stability_analysis.py
│   ├── sensitivity_analysis.py
│   └── control_analysis.py
├── utils/                 # Utilities
│   ├── parameters.py      # Model parameters
│   ├── visualization.py   # Plotting functions
│   └── data_generator.py  # Climate scenarios
├── main.py               # Quick analysis entry
├── run_full_analysis.py  # Full paper analysis
└── .github/workflows/    # GitHub Actions

```

## Workflow Status

[![Climate Analysis](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/climate_analysis.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/climate_analysis.yml)

## Citation

If you use this code in your research, please cite:
```
[Your Paper Title]
[Authors]
[Journal/Conference] [Year]
```

## License

MIT License - see LICENSE file for details.

