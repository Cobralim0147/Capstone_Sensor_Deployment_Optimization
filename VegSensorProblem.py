import numpy as np
from pymoo.core.problem import Problem

class VegSensorProblem(Problem):
    def __init__(self,
                 veg_x: np.ndarray,
                 veg_y: np.ndarray,
                 bed_coords: np.ndarray,
                 sensor_range: float,
                 max_sensors: int,
                 comm_range: float,
                 grid_spacing: float = 1.0):
        """
        Multi-objective optimization problem for sensor placement in agricultural fields.
        
        Parameters:
        -----------
        veg_x, veg_y : np.ndarray
            1D arrays of coordinates for every vegetable plant
        bed_coords : np.ndarray
            (N_beds, 4) array of [x_min, y_min, x_max, y_max] for each raised bed
        sensor_range : float
            Maximum sensing range of sensors
        max_sensors : int
            Maximum number of sensors allowed
        comm_range : float
            Communication range between sensors
        grid_spacing : float
            Spacing between possible sensor positions
        """
        self.veg_pts = np.column_stack([veg_x, veg_y])
        self.bed_coords = bed_coords
        self.sensor_range = sensor_range
        self.comm_range = comm_range
        self.max_sensors = max_sensors
        self.grid_spacing = grid_spacing

        # Build allowed positions for sensor placement
        self.possible_positions = self._build_allowed_positions()
        
        if len(self.possible_positions) == 0:
            raise ValueError("No valid positions found for sensor placement!")
        
        n_var = len(self.possible_positions)
        n_obj = 5  # count, coverage, over-coverage, placement, connectivity
        xl = np.zeros(n_var)
        xu = np.ones(n_var)

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         xl=xl, xu=xu,
                         type_var=int,
                         n_constr=1)

    def _build_allowed_positions(self):
        """
        Build a list of (x,y) coordinates where sensors can be placed.
        Only positions within raised beds are allowed.
        """
        if self.bed_coords.size == 0:
            return np.array([])
        
        # Find the bounds of all beds
        min_x = np.min(self.bed_coords[:, 0])
        max_x = np.max(self.bed_coords[:, 2])
        min_y = np.min(self.bed_coords[:, 1])
        max_y = np.max(self.bed_coords[:, 3])
        
        pts = []
        
        # Create a grid of potential positions
        x_positions = np.arange(min_x, max_x + self.grid_spacing, self.grid_spacing)
        y_positions = np.arange(min_y, max_y + self.grid_spacing, self.grid_spacing)
        
        for x in x_positions:
            for y in y_positions:
                # Check if (x,y) is inside any bed rectangle
                if self._point_in_any_bed(x, y):
                    pts.append((x, y))
        
        return np.array(pts)
    
    def _point_in_any_bed(self, x, y):
        """Check if point (x,y) is inside any bed."""
        for i in range(self.bed_coords.shape[0]):
            x_min, y_min, x_max, y_max = self.bed_coords[i]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate the multi-objective function.
        
        Objectives (all to be minimized):
        1. Sensor count rate (percentage of max sensors used)
        2. Negative coverage rate (to maximize coverage)
        3. Over-coverage rate (minimize redundant coverage)
        4. Negative placement rate (to maximize valid placements)
        5. Negative connectivity rate (to maximize network connectivity)
        """
        pop_size = X.shape[0]
        F = np.zeros((pop_size, self.n_obj))
        G = np.sum(X, axis=1) - self.max_sensors       # ≤ 0 is feasible
        out["G"] = G.reshape((-1,1))


        for i, chromosome in enumerate(X):
            # Decode chromosome: select sensor positions
            selected_indices = chromosome.astype(bool)
            selected_sensors = self.possible_positions[selected_indices]
            n_sensors = selected_sensors.shape[0]

            if n_sensors == 0:
                # No sensors selected - worst case scenario
                F[i] = [0, -0, 100, -100, -0]
                continue

            # Objective 1: Sensor count rate (minimize usage)
            count_rate = 100 * n_sensors / self.max_sensors

            # Objective 2: Coverage rate (maximize plants covered)
            coverage_rate = self._calculate_coverage(selected_sensors)

            # Objective 3: Over-coverage rate (minimize redundant coverage)
            over_coverage_rate = self._calculate_over_coverage(selected_sensors)

            # Objective 4: Placement rate (all selected positions are valid by construction)
            placement_rate = 100.0

            # Objective 5: Connectivity rate (maximize network connectivity)
            connectivity_rate = self._calculate_connectivity(selected_sensors)

            # Store objectives (all to be minimized)
            F[i, 0] = count_rate
            F[i, 1] = -coverage_rate  # Negative to minimize (maximize coverage)
            F[i, 2] = over_coverage_rate
            F[i, 3] = -placement_rate  # Negative to minimize (maximize placement)
            F[i, 4] = -connectivity_rate  # Negative to minimize (maximize connectivity)

        out["F"] = F

    def _calculate_coverage(self, sensors):
        """Calculate the percentage of plants covered by at least one sensor."""
        if sensors.shape[0] == 0 or self.veg_pts.shape[0] == 0:
            return 0.0
        
        # Calculate distances from each sensor to each plant
        distances = np.linalg.norm(
            sensors[:, None, :] - self.veg_pts[None, :, :],
            axis=2
        )  # shape (n_sensors, n_plants)
        
        # Check which plants are covered by at least one sensor
        covered_plants = (distances <= self.sensor_range).any(axis=0)
        coverage_rate = 100 * covered_plants.sum() / self.veg_pts.shape[0]
        
        return coverage_rate

    def _calculate_over_coverage(self, sensors):
        """Calculate the percentage of plants covered by multiple sensors."""
        if sensors.shape[0] == 0 or self.veg_pts.shape[0] == 0:
            return 0.0
        
        # Calculate distances from each sensor to each plant
        distances = np.linalg.norm(
            sensors[:, None, :] - self.veg_pts[None, :, :],
            axis=2
        )
        
        # Count how many sensors cover each plant
        coverage_count = (distances <= self.sensor_range).sum(axis=0)
        over_covered_plants = (coverage_count > 1).sum()
        over_coverage_rate = 100 * over_covered_plants / self.veg_pts.shape[0]
        
        return over_coverage_rate

    def _calculate_connectivity(self, sensors):
        """Calculate the percentage of sensors that are connected to the network."""
        if sensors.shape[0] <= 1:
            return 100.0 if sensors.shape[0] == 1 else 0.0
        
        # Calculate distance matrix between sensors
        distances = np.linalg.norm(
            sensors[:, None, :] - sensors[None, :, :],
            axis=2
        )
        
        # Create adjacency matrix (excluding self-connections)
        adjacency = (distances <= self.comm_range) & (distances > 0)
        
        # Count sensors with at least one connection
        connected_sensors = (adjacency.sum(axis=1) > 0).sum()
        connectivity_rate = 100 * connected_sensors / sensors.shape[0]
        
        return connectivity_rate

    def get_sensor_positions(self, chromosome):
        """Extract actual sensor positions from a chromosome."""
        selected_indices = chromosome.astype(bool)
        return self.possible_positions[selected_indices]

    def evaluate_solution(self, chromosome):
        """Evaluate a single solution and return detailed metrics."""
        sensors = self.get_sensor_positions(chromosome)
        n_sensors = sensors.shape[0]
        
        metrics = {
            'n_sensors': n_sensors,
            'sensor_positions': sensors,
            'count_rate': 100 * n_sensors / self.max_sensors if self.max_sensors > 0 else 0,
            'coverage_rate': self._calculate_coverage(sensors),
            'over_coverage_rate': self._calculate_over_coverage(sensors),
            'placement_rate': 100.0,
            'connectivity_rate': self._calculate_connectivity(sensors)
        }
        
        return metrics