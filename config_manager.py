"""
Automatic configuration optimization based on system capabilities
"""
import psutil
import time

class SmartConfig:
    def __init__(self):
        self.system_memory = psutil.virtual_memory().total / (1024**3)  # GB
        self.cpu_count = psutil.cpu_count()
        
    def get_optimal_settings(self):
        """Automatically determine optimal settings"""
        if self.system_memory < 4:
            return self.low_memory_config()
        elif self.system_memory < 8:
            return self.medium_config()
        else:
            return self.high_performance_config()
    
    def low_memory_config(self):
        return {
            'n_samples_sensitivity': 50,
            'n_samples_mc': 100,
            'simulation_time': 180,
            'quick_mode': True,
            'max_optimization_time': 300  # 5 minutes
        }
    
    def medium_config(self):
        return {
            'n_samples_sensitivity': 200,
            'n_samples_mc': 300,
            'simulation_time': 365,
            'quick_mode': False,
            'max_optimization_time': 900  # 15 minutes
        }
    
    def high_performance_config(self):
        return {
            'n_samples_sensitivity': 500,
            'n_samples_mc': 500,
            'simulation_time': 365,
            'quick_mode': False,
            'max_optimization_time': 1800  # 30 minutes
        }

    def monitor_performance(self, func, timeout=1800):
        """Monitor function execution with timeout"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        try:
            result = func()
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"Execution completed in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            if time.time() - start_time > timeout:
                print(f"Function timed out after {timeout} seconds")
            else:
                print(f"Function failed: {e}")
            return None
