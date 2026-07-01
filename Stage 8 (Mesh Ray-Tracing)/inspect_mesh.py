# Stage 8 (Mesh Ray-Tracing)/inspect_mesh.py
# Run this FIRST — before writing any physics code.
# Confirms mesh scale, watertightness, and orientation.

from pathlib import Path
import numpy as np
import trimesh

# ── Update this if you use .obj instead of .stl ──────────────────────────────
MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone.stl")
# ─────────────────────────────────────────────────────────────────────────────

def inspect_mesh(mesh_path: Path) -> trimesh.Trimesh:
    print("=" * 62)
    print("  STAGE 8 — MESH INSPECTOR")
    print("=" * 62)

    # ── Guard: fail fast with a useful message ────────────────────────────────
    if not mesh_path.exists():
        # Check if they downloaded an .obj instead
        alt_path = mesh_path.with_suffix('.obj')
        if alt_path.exists():
            mesh_path = alt_path
        else:
            raise FileNotFoundError(
                f"\n  ✖  Mesh not found at:\n     {mesh_path}\n\n"
                f"  Please place drone.stl (or drone.obj) in:\n"
                f"     {mesh_path.parent}\n"
            )

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading: {mesh_path.name}")
    mesh = trimesh.load(str(mesh_path), force='mesh')
    print(f"      Vertices : {len(mesh.vertices):>8,}")
    print(f"      Faces    : {len(mesh.faces):>8,}")

    # ── 2. Bounding geometry ──────────────────────────────────────────────────
    print(f"\n[2/5] Bounding Geometry")
    print(f"      Min corner : {mesh.bounds[0]}")
    print(f"      Max corner : {mesh.bounds[1]}")
    print(f"      Extents    : {mesh.extents}  (X=length, Y=span, Z=height)")

    # ── 3. Scale check ────────────────────────────────────────────────────────
    diagonal_mm_threshold = 100   # anything above this is almost certainly mm
    diagonal = float(np.linalg.norm(mesh.extents))
    print(f"\n[3/5] Scale Check")
    print(f"      Bounding diagonal : {diagonal:.3f} units")

    if diagonal > diagonal_mm_threshold:
        print(f"      [WARNING] LIKELY IN MILLIMETRES  (diagonal = {diagonal:.0f} mm ~ "
              f"{diagonal/1000:.2f} m wingspan)")
        print(f"         -> mesh_loader.py must call:  mesh.apply_scale(0.001)")
    elif 0.3 < diagonal < 20.0:
        print(f"      [OK] Scale looks correct (metres). Wingspan ~ {mesh.extents[1]:.2f} m")
    else:
        print(f"      [?] Scale ambiguous (diagonal = {diagonal:.3f}). Inspect visually below.")

    # ── 4. Mesh integrity ─────────────────────────────────────────────────────
    print(f"\n[4/5] Mesh Integrity")
    print(f"      Watertight    : {mesh.is_watertight}")
    print(f"      Volume        : {mesh.volume:.6f} cubic units")
    print(f"      Center of Mass: {mesh.center_mass}")

    if mesh.is_watertight:
        print("      [OK] Watertight — BVH ray-casting is ready.")
    else:
        print("      [WARNING] Not watertight — mesh_loader.py repair pipeline will run.")
        print("         (fill_holes + fix_normals, convex_hull as last resort)")

    # ── 5. Orientation check ──────────────────────────────────────────────────
    print(f"\n[5/5] Orientation Check")
    print(f"      Target convention: Nose -> +X | Right wing -> +Y | Up -> +Z")
    x0, x1 = mesh.bounds[0, 0], mesh.bounds[1, 0]
    y0, y1 = mesh.bounds[0, 1], mesh.bounds[1, 1]
    z0, z1 = mesh.bounds[0, 2], mesh.bounds[1, 2]
    print(f"      X (fore-aft)  : [{x0:+.3f}, {x1:+.3f}]  span = {x1-x0:.3f}")
    print(f"      Y (port-star) : [{y0:+.3f}, {y1:+.3f}]  span = {y1-y0:.3f}  <- wingspan")
    print(f"      Z (belly-dors): [{z0:+.3f}, {z1:+.3f}]  span = {z1-z0:.3f}")
    print()
    print("  If the longest axis is Z (vertical) instead of Y (wingspan),")
    print("  the mesh is standing upright. Rotate in Blender before proceeding:")
    print("  Select all -> R -> X -> -90 -> Enter -> File -> Export -> OBJ")

    print("\n" + "=" * 62)
    print("  Opening 3D viewer ...  (close the window to exit)")
    print("=" * 62 + "\n")
    # mesh.show()

    return mesh


if __name__ == "__main__":
    mesh = inspect_mesh(MESH_PATH)
