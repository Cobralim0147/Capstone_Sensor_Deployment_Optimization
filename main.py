import sys
import os
from pathlib import Path

# Add the deploy_sensors_SPEA2 directory to the path
# This is equivalent to MATLAB's addpath(genpath('deploy_sensors_SPEA2'))
# deploy_sensors_path = Path('deploy_sensors_SPEA2')
# sys.path.append(str(deploy_sensors_path.absolute()))

# Import modules from the path
from environment_generator import environment_generator

def main():
    # 1) Generate environment
    # This is equivalent to [field_map, grid_size, field_length, field_width, monitor_location, vegetable_x, vegetable_y] = environment_generator();
    field_map, grid_size, field_length, field_width, monitor_location, vegetable_x, vegetable_y = environment_generator()
    
    # You can add more code here that uses these variables
    print(f"Field dimensions: {field_length}m x {field_width}m")
    print(f"Grid size: {grid_size}m")
    print(f"Number of vegetable plants: {len(vegetable_x)}")
    print(f"Monitoring room located at: {monitor_location}")
    
    # Return the variables if needed for further processing
    return field_map, grid_size, field_length, field_width, monitor_location, vegetable_x, vegetable_y

if __name__ == "__main__":
    main()