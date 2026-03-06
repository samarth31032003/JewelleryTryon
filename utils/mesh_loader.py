# utils/mesh_loader.py
import trimesh
import numpy as np
from PIL import Image
from utils.logger import logger

log = logger.bind(component="utils")

class MeshData:
    """Simple container for raw mesh data"""
    def __init__(self, vertices, norms, uvs, faces, texture_data=None):
        self.vertices = vertices
        self.norms = norms
        self.uvs = uvs
        self.faces = faces
        self.texture_data = texture_data # RGBA bytes

def load_mesh_data(path, is_occluder=False):
    """Parses 3D file and returns MeshData object."""
    try:
        mesh = trimesh.load(path, force='mesh')
        
        # Calculate the center of mass (centroid)
        center = mesh.centroid
        
        # Move all vertices so the center becomes (0,0,0)
        mesh.vertices -= center
        
        # --- SCALING FIX ---
        # Normalize size so it fits in our view (approx 5 units wide)
        max_span = np.max(mesh.extents)
        if max_span > 0:
            mesh.vertices /= max_span  # Make it size 1.0
            mesh.vertices *= 5.0       # Scale up to size 5.0

        log.info(f"Loaded {path}")
        log.info(f"Original Bounds: {mesh.bounds}")
        mesh.vertices -= mesh.centroid
        log.info(f"New Bounds (Should be centered around 0): {mesh.bounds}")
        
        # 2. Extract Data
        verts = np.array(mesh.vertices, dtype=np.float32)
        
        if hasattr(mesh, 'vertex_normals'):
            norms = np.array(mesh.vertex_normals, dtype=np.float32)
        else:
            norms = np.zeros_like(verts)

        if hasattr(mesh.visual, 'uv') and not is_occluder:
             uvs = np.array(mesh.visual.uv, dtype=np.float32)
        else:
             uvs = np.zeros((len(verts), 2), dtype=np.float32)
             
        faces = np.array(mesh.faces, dtype=np.uint32).flatten()

        # 3. Process Texture (If any)
        texture_bytes = None
        if not is_occluder:
            tex = getattr(mesh.visual.material, 'image', None) or getattr(mesh.visual, 'image', None)
            if tex:
                if not isinstance(tex, Image.Image): 
                    tex = Image.fromarray(tex)
                # Convert to RGBA and Flip Upside Down for OpenGL
                texture_bytes = np.flipud(np.array(tex.convert("RGBA")))

        return MeshData(verts, norms, uvs, faces, texture_bytes)

    except Exception as e:
        log.error(f"Error loading mesh {path}: {e}")
        return None
    
def create_2d_quad(aspect_ratio=1.0):
    """
    Generates a flat 2D plane (Quad) matching the image's aspect ratio.
    """
    # Normalize dimensions so the longest side is exactly 1.0
    if aspect_ratio >= 1.0:
        w = 1.0
        h = 1.0 / aspect_ratio
    else:
        w = aspect_ratio
        h = 1.0

    # 4 Vertices of the rectangle
    vertices = np.array([
        [-w, -h, 0.0], # Bottom-Left
        [ w, -h, 0.0], # Bottom-Right
        [ w,  h, 0.0], # Top-Right
        [-w,  h, 0.0]  # Top-Left
    ], dtype=np.float32)

    # Normals pointing straight at the camera (+Z)
    norms = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # UV Mapping coordinates (how the image wraps onto the rectangle)
    uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    # 2 Triangles to form the Quad
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=np.uint32)

    return MeshData(vertices, norms, uvs, faces, texture_data=None)