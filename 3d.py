import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_topology():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 定义尺寸参数
    # User Request:
    # - Swap Z and X axes. 
    #   - Previous: X=Width, Z=Depth.
    #   - New: Z=Width (Right), X=Depth (Deep). 
    # - Z axis length = 8.5m. X axis length = 5.0m (kept from previous context).
    # - Origin (0,0,0) at bottom-left corner.
    
    room_dim = {'x': 5.0, 'z': 8.5, 'y': 3.5}
    roi_dim = {'x': 2.5, 'z': 2.5, 'y': 2.5}
    rx_height = 0.7

    # Mapping to Matplotlib (MPL)
    # MPL x-axis -> User Z (Width/Right)
    # MPL y-axis -> User X (Depth)
    # MPL z-axis -> User Y (Height/Up)
    
    # Room Ranges (Origin at 0,0,0)
    # User Z: [0, 8.5] -> MPL x
    # User X: [0, 5.0] -> MPL y
    # User Y: [0, 3.5] -> MPL z
    
    # ROI Position
    # Requirement: "Blue cube boundary is 4.5m distance from X axis"
    # The X-axis (User X) is located at Z=0. The distance from it is the Z-coordinate.
    # "Boundary" implies the starting edge (min Z).
    roi_z_start = 4.5
    roi_z_range = [roi_z_start, roi_z_start + roi_dim['z']]
    roi_center_z = roi_z_start + roi_dim['z'] / 2.0
    
    # Centered along X relative to room
    roi_center_x = room_dim['x'] / 2.0
    roi_x_range = [roi_center_x - roi_dim['x']/2, roi_center_x + roi_dim['x']/2]
    
    roi_y_range = [0, roi_dim['y']]

    # Helper to draw box in MPL coords (x=UserZ, y=UserX, z=UserY)
    def draw_box(ax, x_range, y_range, z_range, style, label=None):
        # x_range: [min, max] for MPL x (User Z)
        # y_range: [min, max] for MPL y (User X)
        # z_range: [min, max] for MPL z (User Y)
        
        xx = [x_range[0], x_range[1]]
        yy = [y_range[0], y_range[1]]
        zz = [z_range[0], z_range[1]]
        
        # Bottom (z0)
        ax.plot([xx[0], xx[1], xx[1], xx[0], xx[0]], [yy[0], yy[0], yy[1], yy[1], yy[0]], [zz[0]]*5, **style)
        # Top (z1)
        ax.plot([xx[0], xx[1], xx[1], xx[0], xx[0]], [yy[0], yy[0], yy[1], yy[1], yy[0]], [zz[1]]*5, **style)
        # Verticals
        ax.plot([xx[0], xx[0]], [yy[0], yy[0]], zz, **style)
        ax.plot([xx[1], xx[1]], [yy[0], yy[0]], zz, **style)
        ax.plot([xx[1], xx[1]], [yy[1], yy[1]], zz, **style)
        ax.plot([xx[0], xx[0]], [yy[1], yy[1]], zz, **style)

    # Draw Room
    # MPL X (User Z): [0, 8.5]
    # MPL Y (User X): [0, 5.0]
    draw_box(ax, [0, room_dim['z']], [0, room_dim['x']], [0, room_dim['y']], 
             {'color': 'gray', 'linestyle': '--', 'linewidth': 1}, 'Room')

    # Draw ROI
    draw_box(ax, roi_z_range, roi_x_range, roi_y_range, 
             {'color': 'blue', 'linestyle': ':', 'linewidth': 1.5})

    # Tx Position (Center of ROI on floor?) -> Let's keep it at the center of the NEW ROI
    tx_pos_user = (roi_center_z, roi_center_x, 0) # (Z, X, Y)
    # MPL: (Z, X, Y)
    ax.scatter(tx_pos_user[0], tx_pos_user[1], tx_pos_user[2], c='red', marker='^', s=200, label='Tx', depthshade=False)

    # Rx Positions (Corners of ROI + Midpoints, at h=0.7)
    # ROI Corners in User Z-X plane
    z_min, z_max = roi_z_range
    x_min, x_max = roi_x_range
    
    rx_coords_user = [
        (z_min, x_min), (roi_center_z, x_min), (z_max, x_min),
        (z_max, roi_center_x),
        (z_max, x_max), (roi_center_z, x_max), (z_min, x_max),
        (z_min, roi_center_x)
    ]
    
    for i, (uz, ux) in enumerate(rx_coords_user):
        # MPL coords: x=uz, y=ux, z=rx_height
        ax.scatter(uz, ux, rx_height, c='green', marker='s', s=100, depthshade=False)
        ax.text(uz, ux, rx_height+0.1, f'Rx{i+1}', fontsize=9)
        # Link Tx to Rx
        ax.plot([tx_pos_user[0], uz], [tx_pos_user[1], ux], [0, rx_height], color='orange', linestyle='-', alpha=0.3, linewidth=1)

    # Legend proxy
    ax.scatter([], [], [], c='green', marker='s', s=100, label='Rx Array (h=0.7m)')

    # Labels (Swapped User X/Z)
    ax.set_xlabel('Z (m)') # MPL X
    ax.set_ylabel('X (m)') # MPL Y
    ax.set_zlabel('Y (m)') # MPL Z
    
    # 3. Axis Limits & Aspect Ratio
    ax.set_xlim([0, room_dim['z']])
    ax.set_ylim([0, room_dim['x']])
    ax.set_zlim([0, room_dim['y']])
    
    # Fix the "looks like a cuboid" issue (set 1:1:1 aspect ratio)
    ax.set_box_aspect((room_dim['z'], room_dim['x'], room_dim['y']))

    # Clean styling
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Explicitly mark Origin and Draw Axes to avoid confusion
    # Origin
    ax.scatter(0, 0, 0, color='k', s=50)
    ax.text(0, -0.5, 0, 'Origin (0,0,0)', color='k', fontsize=10)
    
    # Draw Axes Lines manually to verify alignment
    # Z axis (MPL X)
    ax.plot([0,room_dim['z']], [0,0], [0,0], 'k-', lw=1.5)
    # X axis (MPL Y)
    ax.plot([0,0], [0,room_dim['x']], [0,0], 'k-', lw=1.5)
    # Y axis (MPL Z)
    ax.plot([0,0], [0,0], [0,room_dim['y']], 'k-', lw=1.5)


    # View Init:
    # azim=-45 makes origin (0,0) roughly at bottom-left corner with X axis going right, Y axis going deep
    ax.view_init(elev=20, azim=-45)
    plt.legend(loc='right')
    # plt.title("Experimental Topology: Origin at Corner")
    plt.tight_layout()
    plt.show()

draw_topology()
