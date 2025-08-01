# Urban Climate-Social Network Resilience System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Analysis](https://img.shields.io/badge/Analysis-Automated-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

A comprehensive mathematical modeling framework for analyzing the coupled dynamics of climate change, epidemiological spread, and social network evolution in urban environments.

## Overview

This project implements a sophisticated mathematical model that integrates:
- Climate-forced epidemiological dynamics (modified SEIR)
- Dynamic social network evolution
- Coupled system behavior with feedback loops
- Optimal control for resource allocation
- Stability and bifurcation analysis
- Comprehensive sensitivity analysis and uncertainty quantification

## Installation

```bash
git clone https://github.com/yourusername/urban-climate-resilience.git
cd urban-climate-resilience
pip install -r requirements.txt
```

## Usage

### Quick Start

Run a quick baseline analysis:
```bash
python main.py --analysis-type quick --quick-mode
```

### Full Analysis

Run the complete analysis suite:
```bash
python main.py --analysis-type full
```

### Specific Scenarios

Analyze specific climate scenarios:
```bash
# Baseline scenario
python main.py --analysis-type baseline

# Heatwave scenario  
python main.py --analysis-type heatwave

# Extreme climate scenario
python main.py --analysis-type extreme
```

### Command-Line Options

```bash
python main.py [OPTIONS]

Options:
  --analysis-type {full,baseline,heatwave,extreme,quick}
                        Type of analysis to run (default: full)
  --quick-mode          Run in quick mode with reduced parameters
  --parallel            Use parallel processing where available
  --output-dir DIR      Directory for output files (default: results)
```

### Environment Variables

You can customize the analysis using environment variables:
- `QUICK_MODE`: Set to 'true' for quick runs
- `MAX_TIME`: Maximum time in seconds for analysis (default: 3600)
- `N_SAMPLES`: Number of samples for sensitivity analysis (default: 500)
- `SIMULATION_DAYS`: Number of days to simulate (default: 365)

## Project Structure

```
urban-climate-resilience/
├── main.py                 # Main entry point with CLI
├── models/                 # Mathematical models
│   ├── coupled_system.py   # Coupled climate-epidemic-network model
│   ├── epidemiological_model.py  # SEIR model with climate forcing
│   ├── network_model.py    # Dynamic social network model
│   └── optimal_control.py  # Optimal control strategies
├── analysis/               # Analysis modules
│   ├── stability_analysis.py     # Stability and bifurcation analysis
│   ├── sensitivity_analysis.py   # Sensitivity analysis
│   └── control_analysis.py       # Control strategy analysis
├── utils/                  # Utility modules
│   ├── parameters.py       # Model parameters
│   ├── visualization.py    # Plotting and visualization
│   ├── data_generator.py   # Climate scenario generation
│   └── debug.py            # Debug utilities
├── .github/workflows/      # GitHub Actions
│   └── climate_analysis.yml  # Automated analysis workflow
└── test_quick_run.py       # Quick test script
```

## Output Files

The analysis generates the following visualizations in the output directory:

- `epidemic_<scenario>.png` - Epidemic dynamics for each scenario
- `phase_portrait.png` - System phase space analysis
- `bifurcation_diagram.png` - Temperature-dependent stability analysis
- `sensitivity_analysis.png` - Parameter sensitivity results

## GitHub Actions

The repository includes automated analysis via GitHub Actions:
- **Daily runs**: Automated analysis of all scenarios
- **Push triggers**: Analysis runs on code changes
- **Parallel execution**: Scenarios run in parallel for efficiency
- **Artifact storage**: Results stored for 30 days

## Mathematical Models

### 1. Climate-Epidemiological Model
- SEIR compartmental model with climate forcing
- Temperature and humidity dependent transmission rates
- Social behavior modifications under climate stress

### 2. Dynamic Network Model
- Scale-free network with geographic constraints
- Climate-dependent edge formation and dissolution
- Preferential attachment with environmental stress

### 3. Coupled System Dynamics
- Bidirectional coupling between epidemic and network
- Network connectivity affects disease transmission
- Disease prevalence influences social structure

### 4. Optimal Control Framework
- Multi-objective optimization for resource allocation
- Medical, social, and climate mitigation interventions
- Budget constraints and dynamic programming

## Key Results

The model reveals:
- Critical temperature thresholds where system behavior changes dramatically
- Optimal intervention strategies balancing health and social costs
- Early warning indicators for system-wide resilience failures
- Policy recommendations for urban planning and public health

## Applications

- Urban climate adaptation planning
- Public health emergency preparedness
- Social network intervention design
- Resource allocation optimization
- Early warning system development

## License

MIT License - See LICENSE file for details

