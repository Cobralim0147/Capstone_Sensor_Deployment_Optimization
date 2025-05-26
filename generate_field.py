import numpy as np
import matplotlib.pyplot as plt

def environment_generator(field_length, field_width, bed_width, bed_length, furrow_width, grid_size, dot_spacing):
    """
    Generates a field environment with raised beds and furrows, and places vegetables.
    Returns the field map and related parameters.
    
    Parameters:
    -----------
    field_length : float
        Length of the field in meters
    field_width : float
        Width of the field in meters
    bed_width : float
        Width of each raised bed in meters
    bed_length : float
        Length of each raised bed in meters
    furrow_width : float
        Width of furrows between beds in meters
    grid_size : float
        Size of each grid cell in meters
    dot_spacing : float
        Spacing between vegetable plants in meters
    
    Returns:
    --------
    field_map : np.ndarray
        2D array representing the field (0=furrow, 1=bed)
    grid_size : float
        Grid cell size
    field_length : float
        Field length
    field_width : float
        Field width
    monitor_location : np.ndarray
        Coordinates of monitoring room
    vegetable_pos : list
        List of [x, y] coordinates for vegetable plants
    bed_coordinates : np.ndarray
        Array of bed boundaries [x_min, y_min, x_max, y_max]
    """
    
    # Padding (1 furrow width around field)
    padding = furrow_width
    
    # Monitoring room location
    monitor_location = np.array([field_length * 0.9, field_width * 0.1])
    
    # Create grid
    x_cells = int(np.ceil(field_length / grid_size))
    y_cells = int(np.ceil(field_width / grid_size))
    field_map = np.zeros((x_cells, y_cells))  # 0 = furrow, 1 = raised bed
    
    # Convert padding and bed/furrow sizes to cells
    pad_cells = int(np.ceil(padding / grid_size))
    bed_cells = int(np.ceil(bed_width / grid_size))
    furrow_cells = int(np.ceil(furrow_width / grid_size))
    bed_length_cells = int(np.ceil(bed_length / grid_size))
    dot_spacing_cells = max(1, int(np.ceil(dot_spacing / grid_size)))
    
    # Determine usable range inside the field (exclude padding)
    x_start = pad_cells
    x_end = x_cells - pad_cells
    y_start = pad_cells
    y_end = y_cells - pad_cells
    
    # Lists to store vegetable coordinates and bed boundaries
    vegetable_pos = []
    bed_coordinates = []
    
    # Generate beds inside field bounds
    x_pointer = x_start
    while x_pointer + bed_cells <= x_end:
        y_pointer = y_start
        while y_pointer + bed_length_cells <= y_end:
            # Set this area as a raised bed (1)
            field_map[x_pointer:x_pointer + bed_cells, 
                     y_pointer:y_pointer + bed_length_cells] = 1
            
            # Store bed coordinates in real-world units
            bed_x_min = x_pointer * grid_size
            bed_y_min = y_pointer * grid_size
            bed_x_max = (x_pointer + bed_cells) * grid_size
            bed_y_max = (y_pointer + bed_length_cells) * grid_size
            bed_coordinates.append([bed_x_min, bed_y_min, bed_x_max, bed_y_max])
            
            y_pointer += bed_length_cells + furrow_cells
        x_pointer += bed_cells + furrow_cells
    
    # Generate vegetable coordinates inside the beds
    x_pointer = x_start
    while x_pointer + bed_cells <= x_end:
        # Center of the bed in grid coordinates
        x_center = x_pointer + bed_cells / 2
        
        y_pointer = y_start
        while y_pointer + bed_length_cells <= y_end:
            # Place vegetables along the length of the bed
            for y in range(y_pointer + 1, y_pointer + bed_length_cells - 1, dot_spacing_cells):
                x_pos = x_center * grid_size
                y_pos = y * grid_size
                vegetable_pos.append([x_pos, y_pos])
            
            y_pointer += bed_length_cells + furrow_cells
        x_pointer += bed_cells + furrow_cells
    
    return (field_map, grid_size, field_length, field_width, 
            monitor_location, vegetable_pos, np.array(bed_coordinates))


def plot_field(field_map, grid_size, field_length, field_width, 
               monitor_location, vegetable_pos):
    """
    Plot the field layout with beds, furrows, and vegetables.
    """
    vegetable_x = [pos[0] for pos in vegetable_pos]
    vegetable_y = [pos[1] for pos in vegetable_pos]
    
    plt.figure(figsize=(12, 10))
    plt.imshow(field_map.T, extent=[0, field_length, 0, field_width], 
               origin='lower', 
               cmap=plt.cm.colors.ListedColormap([[0.7, 0.7, 0.7], [0.2, 0.5, 0.9]]))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Field Layout: Raised Beds and Furrows (With Vegetables)')
    cbar = plt.colorbar(ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Furrow', 'Bed'])
    
    # Plot vegetables
    if vegetable_x and vegetable_y:
        plt.plot(vegetable_x, vegetable_y, 'g.', markersize=8, label='Vegetables')
    
    # Plot Monitoring Room
    plt.plot(monitor_location[0], monitor_location[1], 'r*', 
             markersize=15, linewidth=2, label='Monitoring Room')
    plt.text(monitor_location[0] + 2, monitor_location[1], 
             'Monitoring Room', color='red', fontsize=10)
    
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()