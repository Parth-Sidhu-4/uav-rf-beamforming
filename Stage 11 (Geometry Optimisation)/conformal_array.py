import numpy as np
import trimesh

def map_antennas_to_mesh(mesh: trimesh.Trimesh, nominal_positions: np.ndarray) -> np.ndarray:
    """
    Snaps nominal antenna positions exactly onto the mesh surface using proximity logic.
    nominal_positions: Nx3 array of (x, y, z) coordinates.
    Returns the exact snapped coordinates.
    """
    closest_points, _, triangle_ids = trimesh.proximity.closest_point(mesh, nominal_positions)
    normals = mesh.face_normals[triangle_ids]
    return closest_points, normals

def generate_nominal_positions() -> np.ndarray:
    """
    Generates 16 nominal antenna positions for the ScanEagle conformal array.
    We'll spread them along the wings (Y-axis) and the fuselage spine (X-axis).
    """
    positions = []
    
    # 4 on the left wing
    positions.append([-0.1, -1.0, 0.05])
    positions.append([-0.1, -0.8, 0.05])
    positions.append([-0.1, -0.6, 0.05])
    positions.append([-0.1, -0.4, 0.05])
    
    # 4 on the right wing
    positions.append([-0.1, 0.4, 0.05])
    positions.append([-0.1, 0.6, 0.05])
    positions.append([-0.1, 0.8, 0.05])
    positions.append([-0.1, 1.0, 0.05])
    
    # 4 on the front fuselage (nose area)
    positions.append([0.2, 0.0, 0.05])
    positions.append([0.4, 0.0, 0.05])
    positions.append([0.6, 0.0, 0.05])
    positions.append([0.8, 0.0, 0.05])
    
    # 4 on the rear fuselage (tail boom)
    positions.append([-0.3, 0.0, 0.05])
    positions.append([-0.5, 0.0, 0.05])
    positions.append([-0.7, 0.0, 0.05])
    positions.append([-0.9, 0.0, 0.05])
    
    return np.array(positions)

def get_conformal_array(mesh: trimesh.Trimesh):
    nominal = generate_nominal_positions()
    snapped, normals = map_antennas_to_mesh(mesh, nominal)
    return snapped, normals

def get_conformal_array_parametric(mesh: trimesh.Trimesh, N: int):
    # Generates N elements, N//4 per zone: left_wing, right_wing, nose, tail_boom
    positions = []
    n_per_zone = max(1, N // 4)
        
    left_y = np.linspace(-1.0, -0.4, n_per_zone)
    right_y = np.linspace(0.4, 1.0, n_per_zone)
    nose_x = np.linspace(0.2, 0.8, n_per_zone)
    tail_x = np.linspace(-0.3, -0.9, n_per_zone)
    
    # Left wing
    for y in left_y:
        positions.append([-0.1, y, 0.05])
    # Right wing
    for y in right_y:
        positions.append([-0.1, y, 0.05])
    # Nose
    for x in nose_x:
        positions.append([x, 0.0, 0.05])
    # Tail
    for x in tail_x:
        positions.append([x, 0.0, 0.05])
        
    positions = np.array(positions)[:N] # Ensure exactly N elements
    snapped, normals = map_antennas_to_mesh(mesh, positions)
    return snapped, normals

def get_semi_distributed_array(mesh: trimesh.Trimesh):
    # 8 elements on right wing, 8 elements on fuselage spine
    positions = []
    
    # 8 right wing
    for y in np.linspace(0.3, 1.0, 8):
        positions.append([-0.1, y, 0.05])
        
    # 8 fuselage spine (nose to tail)
    for x in np.linspace(-0.8, 0.8, 8):
        positions.append([x, 0.0, 0.05])
        
    snapped, normals = map_antennas_to_mesh(mesh, np.array(positions))
    return snapped, normals

def get_clustered_array(mesh: trimesh.Trimesh):
    # 16 elements on dorsal fuselage spine: X-range: -0.3m to +0.3m
    positions = []
    for x in np.linspace(-0.3, 0.3, 16):
        positions.append([x, 0.0, 0.05])
        
    snapped, normals = map_antennas_to_mesh(mesh, np.array(positions))
    return snapped, normals

def get_wingtip_array(mesh, n_per_side=8, tip_margin=0.03, standoff=0.01,
                      y_semispan=1.555):
    chord_fracs = np.linspace(0.15, 0.85, n_per_side // 2)
    positions = []
    
    for side, y_sign in (('left', -1.0), ('right', +1.0)):
        y = y_sign * (y_semispan - tip_margin)
        
        # Dynamically find chord bounds at this Y station
        mask = np.abs(mesh.vertices[:, 1] - y) < 0.05
        verts = mesh.vertices[mask]
        x_lo = verts[:, 0].min()
        x_hi = verts[:, 0].max()
        z_mean = verts[:, 2].mean()
        
        for f in chord_fracs:
            x = x_lo + f * (x_hi - x_lo)
            # Upper element nominal position (Z + 0.2m above tip)
            positions.append([x, y, z_mean + 0.2])
            # Lower element nominal position (Z - 0.2m below tip)
            positions.append([x, y, z_mean - 0.2])
            
    # Snap using closest point
    nominal = np.array(positions)
    closest_pts, _, tri_ids = mesh.nearest.on_surface(nominal)
    normals = mesh.face_normals[tri_ids]
    
    # Apply standoff
    final_pos = closest_pts + normals * standoff
    return final_pos, normals
