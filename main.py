"""
Urban Climate-Social Network Resilience System - Main Entry Point
"""
import os
import sys
import argparse
from pathlib import Path

# Set matplotlib backend before importing
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError as e:
    print(f"Error importing matplotlib: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install: pip install numpy matplotlib")
    sys.exit(1)

# Import local modules
try:
    from models.coupled_system import CoupledSystemModel
    from models.optimal_control import OptimalControlModel
    from analysis.stability_analysis import StabilityAnalysis
    from analysis.sensitivity_analysis import SensitivityAnalysis
    from utils.parameters import ModelParameters
    from utils.visualization import SystemVisualizer
    from utils.data_generator import DataGenerator
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def run_quick_analysis(output_dir='results'):
    """Run a quick analysis for testing"""
    print("Running quick analysis...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Simple parameters
    params = ModelParameters()
    params.N = 1000  # Small population for quick test
    
    # Generate minimal data
    data_gen = DataGenerator(params)
    climate_data, network_data = data_gen.generate_synthetic_data(
        days=30,  # Just 30 days for quick test
        save_path=Path(output_dir) / "quick_data.npz"
    )
    
    # Initialize model
    model = CoupledSystemModel(params, climate_data, network_data)
    
    # Run short simulation
    t_span = (0, 30)
    result = model.simulate(t_span, method='RK45')
    
    # Create simple visualization
    viz = SystemVisualizer(output_dir=output_dir)
    viz.plot_simulation_results(
        result, model,
        title="Quick Test Results",
        save_name="quick_test"
    )
    
    print(f"Analysis complete! Results saved to {output_dir}/")
    print(f"Generated files in {output_dir}:")
    for file in Path(output_dir).glob("*"):
        print(f"  - {file.name}")
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Urban Climate Resilience Analysis')
    parser.add_argument('--analysis-type', type=str, default='quick',
                        choices=['quick', 'baseline', 'heatwave', 'extreme', 'full'],
                        help='Type of analysis to run')
    parser.add_argument('--quick-mode', action='store_true',
                        help='Run in quick mode with reduced parameters')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing')
    
    args = parser.parse_args()
    
    try:
        if args.quick_mode or args.analysis_type == 'quick':
            # Run simplified quick analysis
            success = run_quick_analysis(args.output_dir)
            return 0 if success else 1
        else:
            # For now, just run quick analysis for other modes too
            print(f"Running {args.analysis_type} analysis in quick mode...")
            success = run_quick_analysis(args.output_dir)
            return 0 if success else 1
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
