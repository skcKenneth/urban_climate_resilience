# Urban Climate Resilience Analysis

A mathematical modeling system for analyzing urban climate-social network resilience.

## Quick Start

### Local Installation
```bash
pip install -r requirements.txt
```

### Run Analysis
```bash
python main.py --quick-mode
```

## GitHub Actions

The project includes automated CI/CD that runs on every push to the main branch.

### Workflow Status
[![Climate Analysis](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/climate_analysis.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/climate_analysis.yml)

### Manual Trigger
You can manually trigger the workflow from the Actions tab in GitHub.

## Project Structure
```
├── models/          # Mathematical models
├── analysis/        # Analysis modules
├── utils/           # Utility functions
├── results/         # Output directory (created automatically)
└── main.py          # Main entry point
```

## Output
The analysis generates visualizations saved as PNG files in the `results/` directory.

