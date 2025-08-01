# Combined Climate Analysis Results - Fri Aug  1 11:35:23 UTC 2025

## Summary

This directory contains the complete results from the automated climate-epidemic analysis run.

## Analysis Components

### 1. Baseline System Analysis
- Status: success
- Focus: Core system dynamics without interventions

### 2. Sensitivity Analysis  
- Status: success
- Focus: Parameter sensitivity and system robustness

### 3. Uncertainty Quantification
- Status: success
- Focus: Monte Carlo uncertainty analysis

### 4. Optimal Control Analysis
- Status: success
- Focus: Optimal resource allocation strategies

## Directory Structure

```
2025-08-01/
├── data/          # Analysis results in JSON/CSV format
├── figures/       # All generated visualizations
└── logs/          # Execution logs
```

## Key Visualizations

### Control Strategy Comparison
![Strategy Dashboard](figures/strategy_dashboard.png)

### Optimal Control Trajectories
![Control Trajectories](figures/control_trajectories.png)

### 3D Phase Space
![Phase Space](figures/phase_space_3d.png)

### Sensitivity Analysis
![Sensitivity Heatmap](figures/sensitivity_heatmap.png)

### Uncertainty Analysis
![Uncertainty Bands](figures/uncertainty_bands.png)

## Run Information
- Date: 2025-08-01T11:35:23Z
- Run Number: 6
- Commit: a84a2f9b063ea8bf0f3d569312f1864196c4545c
- Triggered by: workflow_dispatch

## Files Generated

### Data Files
```
total 24
drwxr-xr-x 2 runner docker 4096 Aug  1 11:35 .
drwxr-xr-x 5 runner docker 4096 Aug  1 11:35 ..
-rw-r--r-- 1 runner docker  520 Aug  1 11:35 baseline_summary.json
-rw-r--r-- 1 runner docker  519 Aug  1 11:35 control_summary.json
-rw-r--r-- 1 runner docker  523 Aug  1 11:35 sensitivity_summary.json
-rw-r--r-- 1 runner docker  523 Aug  1 11:35 uncertainty_summary.json
```

### Figures
```
total 2760
drwxr-xr-x 2 runner docker   4096 Aug  1 11:35 .
drwxr-xr-x 5 runner docker   4096 Aug  1 11:35 ..
-rw-r--r-- 1 runner docker 263261 Aug  1 11:35 bifurcation_diagram.png
-rw-r--r-- 1 runner docker 654788 Aug  1 11:35 epidemic_dynamics.png
-rw-r--r-- 1 runner docker 556087 Aug  1 11:35 epidemic_dynamics_baseline.png
-rw-r--r-- 1 runner docker 552212 Aug  1 11:35 epidemic_dynamics_extreme.png
-rw-r--r-- 1 runner docker 563066 Aug  1 11:35 epidemic_dynamics_heatwave.png
-rw-r--r-- 1 runner docker 219968 Aug  1 11:35 phase_portrait.png
```

### Logs
```
total 24
drwxr-xr-x 2 runner docker 4096 Aug  1 11:35 .
drwxr-xr-x 5 runner docker 4096 Aug  1 11:35 ..
-rw-r--r-- 1 runner docker 1404 Aug  1 11:35 baseline_log.txt
-rw-r--r-- 1 runner docker 1403 Aug  1 11:35 control_log.txt
-rw-r--r-- 1 runner docker 1408 Aug  1 11:35 sensitivity_log.txt
-rw-r--r-- 1 runner docker 1407 Aug  1 11:35 uncertainty_log.txt
```
