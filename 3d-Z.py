import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_sampling_cube():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Dimensions
    cube_size = 2.5
    
    # Coordinate System Strategy:
    # Origin (0,0,0) is at the Bottom-Right-Front corner of the cube.
    # User X axis: Horizontal Right. 
    #   Since Origin is at Right, and space is to the Left, User X coordinates will be [-2.5, 0].
    # User Y axis: Vertical Up. [0, 2.5].
    # User Z axis: Horizontal Deep (Into screen). [0, 2.5].
    
    # Mapping to Matplotlib (MPL) Axes to achieve the visual look:
    # MPL X corresponds to User X (Right)
    # MPL Y corresponds to User Z (Deep)
    # MPL Z corresponds to User Y (Up)
    
    # Define ranges
    x_range = [-cube_size, 0] # User X
    y_range = [0, cube_size]  # User Z (Deep)
    z_range = [0, cube_size]  # User Y
    
    # 1. Draw Cube Wireframe (Blue)
    def draw_wireframe_box(ax, xr, yr, zr, color, style, label):
        # Bottom face
        ax.plot([xr[0], xr[1], xr[1], xr[0], xr[0]], [yr[0], yr[0], yr[1], yr[1], yr[0]], [zr[0]]*5, color=color, linestyle=style, label=label)
        # Top face
        ax.plot([xr[0], xr[1], xr[1], xr[0], xr[0]], [yr[0], yr[0], yr[1], yr[1], yr[0]], [zr[1]]*5, color=color, linestyle=style)
        # Verticals
        ax.plot([xr[0], xr[0]], [yr[0], yr[0]], zr, color=color, linestyle=style)
        ax.plot([xr[1], xr[1]], [yr[0], yr[0]], zr, color=color, linestyle=style)
        ax.plot([xr[1], xr[1]], [yr[1], yr[1]], zr, color=color, linestyle=style)
        ax.plot([xr[0], xr[0]], [yr[1], yr[1]], zr, color=color, linestyle=style)

    draw_wireframe_box(ax, x_range, y_range, z_range, 'blue', '-', 'ROI Boundary')

    # 2. Draw Sampling Points
    # Z planes (User Z -> MPL Y)
    z_planes = [0.5, 1.5, 2.5]
    z_labels = ['Near (0.5m)', 'Mid (1.5m)', 'Far (2.5m)']
    
    # In each plane: 3x3 array (Center, 4 Corners, 4 Midpoints)
    # User X coords: 0, -1.25, -2.5 (mapped to MPL X)
    # User Y coords: 0, 1.25, 2.5 (mapped to MPL Z)
    
    user_x_vals = [0, -1.25, -2.5] 
    user_y_vals = [0, 1.25, 2.5]
    
    colors = ['red', 'green', 'orange']
    
    for i, z_val in enumerate(z_planes):
        # Create grid points
        # MPL Y is constant for this plane (User Z)
        mpl_y = z_val 
        
        # Grid arrays
        xs = []
        ys = [] # MPL Y (User Z)
        zs = [] # MPL Z (User Y)
        
        for ux in user_x_vals:
            for uy in user_y_vals:
                xs.append(ux)
                ys.append(mpl_y)
                zs.append(uy)
                
        # Plot points
        ax.scatter(xs, ys, zs, c=colors[i], s=50, label=f'Plane Z={z_val}m')
        
        # Draw the plane rectangle for visual reference (transparent)
        verts = [
            [(x_range[0], mpl_y, z_range[0]),
             (x_range[1], mpl_y, z_range[0]),
             (x_range[1], mpl_y, z_range[1]),
             (x_range[0], mpl_y, z_range[1])]
        ]
        poly = Poly3DCollection(verts, alpha=0.1, facecolor=colors[i])
        ax.add_collection3d(poly)

    # 3. Setup Coordinate System Labels and Origin
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)') # MPL Y is User Z
    ax.set_zlabel('Y (m)') # MPL Z is User Y
    
    # Mark Origin
    ax.scatter(0, 0, 0, c='black', s=100, marker='o')
    ax.text(0.1, -0.1, 0, 'Origin', color='black', weight='bold')

    # Limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    
    # Aspect Ratio 1:1:1
    ax.set_box_aspect((cube_size, cube_size, cube_size))
    
    # View Angle
    ax.view_init(elev=20, azim=-60)
    
    plt.title("Layered Sampling Scheme (3x3 Grid per Plane)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

draw_sampling_cube()
