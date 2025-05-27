import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# Import your custom modules
from generate_field import environment_generator, plot_field
from VegSensorProblem import VegSensorProblem

def visualize_solution(problem, solution_chromosome, field_length, field_width, 
                      vegetable_pos, bed_coords, title="Sensor Placement Solution"):
    """
    Visualize a sensor placement solution.
    """
    # Get sensor positions
    sensors = problem.get_sensor_positions(solution_chromosome)
    metrics = problem.evaluate_solution(solution_chromosome)
    
    # Extract vegetable coordinates
    veg_x = [pos[0] for pos in vegetable_pos]
    veg_y = [pos[1] for pos in vegetable_pos]
    
    plt.figure(figsize=(12, 10))
    
    # Plot bed boundaries
    for bed in bed_coords:
        x_min, y_min, x_max, y_max = bed
        plt.plot([x_min, x_max, x_max, x_min, x_min], 
                [y_min, y_min, y_max, y_max, y_min], 
                'b-', linewidth=2, alpha=0.7)
    
    # Plot vegetables
    if veg_x and veg_y:
        plt.plot(veg_x, veg_y, 'g.', markersize=6, alpha=0.7, label='Vegetables')
    
    # Plot sensors
    if sensors.shape[0] > 0:
        plt.plot(sensors[:, 0], sensors[:, 1], 'ro', markersize=10, 
                label=f'Sensors ({sensors.shape[0]})')
        
        # Plot sensor coverage circles
        for sensor in sensors:
            circle = plt.Circle(sensor, problem.sensor_range, 
                              fill=False, color='red', alpha=0.3, linestyle='--')
            plt.gca().add_patch(circle)
        
        # Plot communication links
        if sensors.shape[0] > 1:
            distances = np.linalg.norm(
                sensors[:, None, :] - sensors[None, :, :], axis=2
            )
            for i in range(sensors.shape[0]):
                for j in range(i + 1, sensors.shape[0]):
                    if distances[i, j] <= problem.comm_range:
                        plt.plot([sensors[i, 0], sensors[j, 0]], 
                                [sensors[i, 1], sensors[j, 1]], 
                                'b--', alpha=0.5, linewidth=1)
    
    plt.xlim(0, field_length)
    plt.ylim(0, field_width)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'{title}\n'
              f'Coverage: {metrics["coverage_rate"]:.1f}%, '
              f'Sensors: {metrics["n_sensors"]}, '
              f'Connectivity: {metrics["connectivity_rate"]:.1f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def analyze_pareto_front(res, problem, top_n=5):
    """
    Analyze the Pareto front and return top solutions.
    """
    if res.X is None:
        print("No solutions found!")
        return []
    
    # Get objective values
    F = res.F
    X = res.X
    
    print(f"\nPareto Front Analysis:")
    print(f"Number of solutions: {len(X)}")
    print(f"Objective ranges:")
    print(f"  Count rate: [{F[:, 0].min():.1f}, {F[:, 0].max():.1f}]")
    print(f"  Coverage rate: [{-F[:, 1].max():.1f}, {-F[:, 1].min():.1f}]")
    print(f"  Over-coverage: [{F[:, 2].min():.1f}, {F[:, 2].max():.1f}]")
    print(f"  Connectivity: [{-F[:, 4].max():.1f}, {-F[:, 4].min():.1f}]")
    
    # Find interesting solutions
    solutions = []
    
    # Best coverage solution
    best_coverage_idx = np.argmin(F[:, 1])  # Most negative = best coverage
    solutions.append(("Best Coverage", best_coverage_idx, X[best_coverage_idx]))
    
    # Most efficient solution (good coverage with fewer sensors)
    efficiency_score = -F[:, 1] / (F[:, 0] + 1e-6)  # coverage / count
    best_efficiency_idx = np.argmax(efficiency_score)
    solutions.append(("Most Efficient", best_efficiency_idx, X[best_efficiency_idx]))
    
    # Best connectivity solution
    best_connectivity_idx = np.argmin(F[:, 4])  # Most negative = best connectivity
    solutions.append(("Best Connectivity", best_connectivity_idx, X[best_connectivity_idx]))
    
    # Print solution details
    for name, idx, chromosome in solutions:
        metrics = problem.evaluate_solution(chromosome)
        print(f"\n{name} Solution:")
        print(f"  Sensors: {metrics['n_sensors']}")
        print(f"  Coverage: {metrics['coverage_rate']:.1f}%")
        print(f"  Over-coverage: {metrics['over_coverage_rate']:.1f}%")
        print(f"  Connectivity: {metrics['connectivity_rate']:.1f}%")
    
    return solutions

def main():
    """
    Main optimization function.
    """
    print("Starting sensor placement optimization...")
    
    # 1) Generate field environment
    print("Generating field environment...")
    results = environment_generator(
        field_length=100,
        field_width=100,
        bed_width=0.8,
        bed_length=5,  # Increased bed length
        furrow_width=1.5,
        grid_size=0.5,
        dot_spacing=0.4  # Increased spacing
    )
    
    field_map, grid_size, field_length, field_width, monitor_location, vegetable_pos, bed_coords = results
    
    print(f"Field generated:")
    print(f"  Dimensions: {field_length}m x {field_width}m")
    print(f"  Number of beds: {len(bed_coords)}")
    print(f"  Number of vegetables: {len(vegetable_pos)}")
    
    # Plot the field
    # plot_field(field_map, grid_size, field_length, field_width, 
    #           monitor_location, vegetable_pos)
    
    # 2) Create optimization problem
    print("Setting up optimization problem...")
    
    veg_x = np.array([pos[0] for pos in vegetable_pos])
    veg_y = np.array([pos[1] for pos in vegetable_pos])
    
    problem = VegSensorProblem(
        veg_x=veg_x,
        veg_y=veg_y,
        bed_coords=bed_coords,
        sensor_range=1.5,    # Sensor range in meters
        max_sensors=50,      # Maximum number of sensors
        comm_range=10.0,     # Communication range in meters
        grid_spacing=0.5     # Grid spacing for sensor positions
    )
    
    print(f"Problem setup:")
    print(f"  Possible sensor positions: {len(problem.possible_positions)}")
    print(f"  Decision variables: {problem.n_var}")
    print(f"  Objectives: {problem.n_obj}")
    
    # 3) Run optimization
    print("Running SPEA-II optimization...")
    
    algorithm = SPEA2(
        pop_size=30,        # Population size
        archive_size=20      # Archive size
    )
    
    termination = get_termination("n_gen", 20)
    
    res = minimize(
        problem,
        algorithm,
        termination,
        verbose=True
    )
    
    print("Optimization complete!")
    
    # 4) Analyze results
    if res.X is not None:
        solutions = analyze_pareto_front(res, problem)
        
        # Visualize top solutions
        for name, idx, chromosome in solutions[:2]:  # Show top 2 solutions
            visualize_solution(problem, chromosome, field_length, field_width,
                             vegetable_pos, bed_coords, title=f"{name} Solution")
    else:
        print("No feasible solutions found!")
    
    return res, problem

if __name__ == "__main__":
    try:
        result, problem = main()
        print("\nOptimization completed successfully!")
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()