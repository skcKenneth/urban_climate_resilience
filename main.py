#!/usr/bin/env python3
"""
Minimal working version for testing
"""
import os
import sys

# Set matplotlib backend first
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("Starting minimal climate analysis...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Generate simple data
    days = 30
    time = np.linspace(0, days, days)
    temperature = 25 + 5 * np.sin(2 * np.pi * time / 365) + np.random.normal(0, 1, days)
    infected = 100 * np.exp(-0.1 * time) + np.random.normal(0, 5, days)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Temperature plot
    ax1.plot(time, temperature, 'r-', linewidth=2)
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Climate Analysis - Temperature Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Infected plot
    ax2.plot(time, infected, 'b-', linewidth=2)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Infected Population')
    ax2.set_title('Epidemic Dynamics')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/climate_analysis.png', dpi=150)
    plt.close()
    
    # Save data
    np.savez('results/data.npz', time=time, temperature=temperature, infected=infected)
    
    # Create summary
    with open('results/summary.txt', 'w') as f:
        f.write("Climate Analysis Summary\n")
        f.write("=======================\n")
        f.write(f"Days simulated: {days}\n")
        f.write(f"Avg temperature: {np.mean(temperature):.1f}°C\n")
        f.write(f"Total infected: {np.sum(infected):.0f}\n")
    
    print("Analysis complete!")
    print("Files generated:")
    for file in os.listdir('results'):
        print(f"  - {file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
