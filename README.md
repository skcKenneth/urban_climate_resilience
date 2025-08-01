# Climate-Epidemic Coupled System Analysis

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/your-username/climate-epidemic-model/actions/workflows/run_analysis.yml/badge.svg)](https://github.com/your-username/climate-epidemic-model/actions)

A computational framework for analyzing the coupled dynamics of climate change and epidemic spread, incorporating social network effects and optimal control strategies.

## Overview

This repository contains the implementation of a coupled climate-epidemic model that examines:
- SEIR epidemic dynamics under climate influence
- Temperature-dependent transmission rates
- Network structure evolution during epidemics
- Optimal control strategies for epidemic mitigation
- Sensitivity and uncertainty quantification

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/climate-epidemic-model.git
cd climate-epidemic-model

# Install dependencies
pip install -r requirements.txt

# Run the main analysis
python run_analysis.py

# Generate all figures
python generate_all_figures.py
```

## Model Description

### Epidemic Model (SEIR)
The model uses a modified SEIR framework with climate coupling:
- **S**: Susceptible population
- **E**: Exposed population  
- **I**: Infected population
- **R**: Recovered population

### Climate Coupling
Temperature affects transmission rate through:
```
β(T) = β₀ * (1 + α_T * (T - T_ref))
```

### Network Dynamics
Contact network evolves based on:
- Disease awareness
- Climate conditions
- Social distancing measures

## Repository Structure

```
climate-epidemic-model/
├── models/                 # Core model implementations
│   ├── epidemic_model.py   # SEIR dynamics
│   ├── climate_model.py    # Climate scenarios
│   └── coupled_model.py    # Integrated system
├── analysis/              # Analysis modules
│   ├── sensitivity_analysis.py
│   └── control_analysis.py
├── utils/                 # Utilities
│   └── visualization.py   # Plotting functions
├── figures/               # Generated figures
├── run_analysis.py        # Main analysis script
└── requirements.txt       # Dependencies
```

## Key Results

The analysis produces:

1. **Epidemic Dynamics**: Comparison across climate scenarios (baseline, heatwave, extreme)
2. **Phase Portraits**: System trajectories in S-I phase space
3. **Sensitivity Analysis**: Parameter importance via Sobol indices
4. **Control Strategies**: Comparison of intervention effectiveness

## Usage Examples

### Run Basic Simulation
```python
from models.epidemic_model import EpidemicModel
from models.climate_model import ClimateModel
from models.coupled_model import CoupledClimateEpidemicNetwork

# Initialize models
epidemic = EpidemicModel()
climate = ClimateModel()
coupled = CoupledClimateEpidemicNetwork(epidemic, climate)

# Run simulation
t = np.linspace(0, 365, 365)
results = coupled.simulate(t)
```

### Generate Custom Figures
```python
from utils.visualization import Visualizer

viz = Visualizer()
viz.plot_epidemic_dynamics(results, save_path='my_figure.png')
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2024climate,
  title={Coupled Climate-Epidemic Dynamics with Network Effects},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## Requirements

- Python 3.8+
- NumPy
- SciPy  
- Matplotlib
- Seaborn
- NetworkX

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contact

For questions or collaborations, please contact [your-email@example.com]

