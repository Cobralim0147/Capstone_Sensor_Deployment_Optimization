import numpy as np
import matplotlib.pyplot as plt

def environment_generator():
    """
    Generates a field environment with raised beds and furrows, and places vegetables.
    Returns the field map and related parameters.
    """
    # Parameters
    field_length = 100  # meters
    field_width = 100  # meters
    grid_size = 0.5  # meters (each grid cell is 0.5m x 0.5m)
    
    # Raised Bed and Furrow Dimensions
    bed_width = 0.8  # meters
    furrow_width = 1.5  # meters
    bed_length = 5  # meters
    dot_spacing = 0.4  # meters
    
    # Padding (1 furrow width around field)
    padding = furrow_width  # meters
    
    # Monitoring room location
    monitor_location = np.array([150, 50])
    
    # Create grid
    x_cells = round(field_length / grid_size)
    y_cells = round(field_width / grid_size)
    field_map = np.zeros((x_cells, y_cells))  # 0 = furrow, 1 = raised bed
    
    # Convert padding and bed/furrow sizes to cells
    pad_cells = round(padding / grid_size)
    bed_cells = round(bed_width / grid_size)
    furrow_cells = round(furrow_width / grid_size)
    bed_length_cells = round(bed_length / grid_size)
    dot_spacing_cells = round(dot_spacing / grid_size)
    
    # Determine usable range inside the field (exclude padding)
    x_start = pad_cells + 1
    x_end = x_cells - pad_cells
    y_start = pad_cells + 1
    y_end = y_cells - pad_cells
    
    # Lists to store vegetable coordinates
    vegetable_x = []
    vegetable_y = []
    
    # Generate beds inside field bounds
    x_pointer = x_start
    while x_pointer + bed_cells - 1 <= x_end:
        # Vertical cycle
        for x in range(x_pointer, x_pointer + bed_cells):
            y_pointer = y_start
            while y_pointer + bed_length_cells - 1 <= y_end:
                # Set this area as a raised bed (1)
                field_map[x-1:x, y_pointer-1:y_pointer + bed_length_cells] = 1
                y_pointer = y_pointer + bed_length_cells + furrow_cells
        x_pointer = x_pointer + bed_cells + furrow_cells
    
    # Generate vegetable coordinates inside the beds
    x_pointer = x_start
    while x_pointer + bed_cells - 1 <= x_end:
        x_center = x_pointer + int(bed_cells / 2)
        y_pointer = y_start
        while y_pointer + bed_length_cells - 1 <= y_end:
            for y in range(y_pointer + 1, y_pointer + bed_length_cells, dot_spacing_cells):
                # Convert cell coordinates to meters
                x_pos = (x_center - 0.5) * grid_size
                y_pos = (y - 0.5) * grid_size
                vegetable_x.append(x_pos)
                vegetable_y.append(y_pos)
            y_pointer = y_pointer + bed_length_cells + furrow_cells
        x_pointer = x_pointer + bed_cells + furrow_cells
    
    # Plot the Field
    plt.figure(figsize=(10, 8))
    plt.imshow(field_map.T, extent=[0, field_length, 0, field_width], origin='lower', 
              cmap=plt.cm.colors.ListedColormap([[0.7, 0.7, 0.7], [0.2, 0.5, 0.9]]))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Field Layout: Raised Beds and Furrows (With Vegetables)')
    cbar = plt.colorbar(ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Furrow', 'Bed'])
    
    # Plot vegetables
    plt.plot(vegetable_x, vegetable_y, 'g.', markersize=12)
    
    # Plot Monitoring Room
    plt.plot(monitor_location[0], monitor_location[1], 'r*', markersize=12, linewidth=2)
    plt.text(monitor_location[0]+2, monitor_location[1], 'Monitoring Room', color='red', fontsize=10)
    
    plt.axis('equal')
    plt.grid(True)
    plt.show()
    
    return field_map, grid_size, field_length, field_width, monitor_location, vegetable_x, vegetable_y

