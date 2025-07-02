import numpy as np 

class CoverageCalculator:
    """Utility class for coverage calculations."""
    
    @staticmethod
    def calculate_coverage_rate(sensors: np.ndarray, plants: np.ndarray, sensor_range: float) -> float:
        """Calculate the percentage of plants covered by at least one sensor."""
        if sensors.size == 0 or plants.size == 0:
            return 0.0
        
        try:
            # Calculate distances from each sensor to each plant
            distances = np.linalg.norm(
                sensors[:, None, :] - plants[None, :, :],
                axis=2
            )
            
            # Check which plants are covered by at least one sensor
            covered_plants = (distances <= sensor_range).any(axis=0)
            coverage_rate = 100.0 * covered_plants.sum() / plants.shape[0]
            
            return float(coverage_rate)
            
        except Exception as e:
            print(f"Error calculating coverage rate: {e}")
            return 0.0
    
    @staticmethod
    def calculate_over_coverage_rate(sensors: np.ndarray, plants: np.ndarray, sensor_range: float) -> float:
        """Calculate the percentage of plants covered by multiple sensors."""
        if sensors.size == 0 or plants.size == 0:
            return 0.0
        
        try:
            # Calculate distances from each sensor to each plant
            distances = np.linalg.norm(
                sensors[:, None, :] - plants[None, :, :],
                axis=2
            )
            
            # Count how many sensors cover each plant
            coverage_count = (distances <= sensor_range).sum(axis=0)
            over_covered_plants = (coverage_count > 1).sum()
            over_coverage_rate = 100.0 * over_covered_plants / plants.shape[0]
            
            return float(over_coverage_rate)
            
        except Exception as e:
            print(f"Error calculating over-coverage rate: {e}")
            return 0.0
    
    @staticmethod
    def calculate_connectivity_rate(sensors: np.ndarray, comm_range: float) -> float:
        """Calculate the percentage of sensors that are connected to the network."""
        if sensors.size == 0:
            return 0.0
        if sensors.shape[0] == 1:
            return 100.0
        
        try:
            # Calculate distance matrix between sensors
            distances = np.linalg.norm(
                sensors[:, None, :] - sensors[None, :, :],
                axis=2
            )
            
            # Create adjacency matrix (excluding self-connections)
            adjacency = (distances <= comm_range) & (distances > 0)
            
            # Count sensors with at least one connection
            connected_sensors = (adjacency.sum(axis=1) > 0).sum()
            connectivity_rate = 100.0 * connected_sensors / sensors.shape[0]
            
            return float(connectivity_rate)
            
        except Exception as e:
            print(f"Error calculating connectivity rate: {e}")
            return 0.0
