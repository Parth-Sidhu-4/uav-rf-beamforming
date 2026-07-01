import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Stage 8 (Mesh Ray-Tracing)')))
from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array

def run_histogram():
    mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
    
    # We just need to extract silhouette edges for a typical jammer direction
    jammer_dir_body = np.array([0, 1, 0])
    
    face_normals = mesh.face_normals
    dot_prod = np.dot(face_normals, jammer_dir_body)
    
    front_facing = dot_prod > 1e-6
    back_facing = dot_prod < -1e-6
    
    adj = mesh.face_adjacency
    edges = mesh.face_adjacency_edges
    
    is_silhouette = (front_facing[adj[:, 0]] & back_facing[adj[:, 1]]) | \
                    (back_facing[adj[:, 0]] & front_facing[adj[:, 1]])
                    
    sil_edges = edges[is_silhouette]
    V1 = mesh.vertices[sil_edges[:, 0]]
    V2 = mesh.vertices[sil_edges[:, 1]]
    
    edge_lengths = np.linalg.norm(V1 - V2, axis=1)
    
    plt.figure(figsize=(10, 6))
    # We want to see the distribution of lengths up to 10 cm
    plt.hist(edge_lengths * 100, bins=200, range=(0, 10), color='blue', alpha=0.7)
    plt.axvline(1.5, color='red', linestyle='dashed', linewidth=2, label='Proposed 1.5 cm filter')
    plt.title('Distribution of Silhouette Edge Lengths')
    plt.xlabel('Edge Length (cm)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('diagnostic_histogram.png')
    print("Saved diagnostic_histogram.png")
    
if __name__ == "__main__":
    run_histogram()
