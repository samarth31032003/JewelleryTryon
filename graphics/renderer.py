# graphics/renderer.py
import ctypes
import numpy as np
import cv2
import os
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PyQt5.QtWidgets import QOpenGLWidget
from utils.logger import logger

log = logger.bind(component="ui")

from graphics.shaders import *
from utils.mesh_loader import load_mesh_data, create_2d_quad
from utils.paths import OCCLUDER_HEAD_PATH, OCCLUDER_BODY_PATH

class ARViewerWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.meshes = {} 
        self.occluder_meshes = {} 
        
        # State Flags
        self.camera_ready = False
        self.show_grid = False
        self.debug_occluder = False 
        self.cylinder_ready = False

        # GL IDs
        self.cam_tex_id = None
        self.vao_bg = None
        self.vao_debug = None
        self.vao_gizmo = None
        
        # Occluder (Procedural Cylinder)
        self.vao_cylinder = None
        self.idx_count_cylinder = 0
        
        # Render Lists
        self.render_instances = []   
        self.occluder_instances = [] 
        
        # Scene Settings
        self.fov = 40.0
        self.near_plane = 0.1
        self.far_plane = 1000.0
        self.light_pos = [0.0, 10.0, 10.0]

        # custom lighting
        self.ambient_str = 0.4   # Keep shadows dark or 1.0 to reset
        self.diffuse_str = 0.6   # 0.4 + 0.6 = 1.0 (Prevents blowout) or 0.0 to reset
        self.exposure = 1.0
        self.gamma = 1.0

        # Camera / Viewport
        self.cam_w = 640
        self.cam_h = 480
        self.img_aspect = 640/480
        self.viewport_rect = (0, 0, 640, 480)
        
        self.proj = np.eye(4, dtype=np.float32)
        self.view = np.eye(4, dtype=np.float32)
        
        self.debug_count = 0

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
        try:
            self.prog_mesh = compileProgram(compileShader(MESH_VS, GL_VERTEX_SHADER), compileShader(MESH_FS, GL_FRAGMENT_SHADER))
            self.prog_bg = compileProgram(compileShader(BG_VS, GL_VERTEX_SHADER), compileShader(BG_FS, GL_FRAGMENT_SHADER))
            self.prog_debug = compileProgram(compileShader(DEBUG_VS, GL_VERTEX_SHADER), compileShader(DEBUG_FS, GL_FRAGMENT_SHADER))
            self.prog_occ = compileProgram(compileShader(MESH_VS, GL_VERTEX_SHADER), compileShader(OCCLUDER_FS, GL_FRAGMENT_SHADER))
        except Exception as e:
            log.error(f"Shader Compile Error: {e}")

        # Cache Uniforms
        self.loc_m_model = glGetUniformLocation(self.prog_mesh, "u_model")
        self.loc_m_view = glGetUniformLocation(self.prog_mesh, "u_view")
        self.loc_m_proj = glGetUniformLocation(self.prog_mesh, "u_proj")
        self.loc_m_tex = glGetUniformLocation(self.prog_mesh, "u_tex")
        self.loc_m_has_tex = glGetUniformLocation(self.prog_mesh, "u_has_tex")
        self.loc_m_color = glGetUniformLocation(self.prog_mesh, "u_color_override")
        self.loc_l_pos = glGetUniformLocation(self.prog_mesh, "u_light_pos")
        self.loc_l_amb = glGetUniformLocation(self.prog_mesh, "u_ambient_str")
        self.loc_l_diff = glGetUniformLocation(self.prog_mesh, "u_diffuse_str")
        
        # Debug Uniforms
        self.loc_d_model = glGetUniformLocation(self.prog_debug, "u_model")
        self.loc_d_view = glGetUniformLocation(self.prog_debug, "u_view")
        self.loc_d_proj = glGetUniformLocation(self.prog_debug, "u_proj")
        
        self._init_bg_quad()
        self._init_debug_layer() 
        self.init_occluder_primitive()
        
        # --- 2. AUTO-LOAD OCCLUDERS (Context is guaranteed valid here) ---
        log.info("[Renderer] Initializing Occluders...")
        self.load_occluder(str(OCCLUDER_HEAD_PATH), "occ_head")
        self.load_occluder(str(OCCLUDER_BODY_PATH), "occ_body")

    # --- MEMORY MANAGEMENT ---
    def clear_scene(self):
        """Frees GPU memory for JEWELRY ONLY. Keeps Occluders."""
        self.makeCurrent()
        # Only delete Jewelry Meshes
        for key, mesh_data in self.meshes.items():
            if mesh_data['tex']: glDeleteTextures([mesh_data['tex']])
            if mesh_data['vao']: glDeleteVertexArrays(1, [mesh_data['vao']])
        self.meshes = {}
        # NOT clearing self.occluder_meshes here anymore.
        self.doneCurrent()

    # --- HELPER FUNCTIONS ---
    def _init_bg_quad(self):
        data = np.array([-1,-1,0,1, 1,-1,1,1, -1,1,0,0, 1,1,1,0], dtype=np.float32)
        self.vao_bg = glGenVertexArrays(1); glBindVertexArray(self.vao_bg)
        vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

    def _init_debug_layer(self):
        # Grid Lines
        lines = []; colors = []
        size = 50; step = 5; grid_y = -10
        for i in range(-size, size + 1, step):
            lines.extend([i, grid_y, -size, i, grid_y, size])
            colors.extend([0.3]*6)
            lines.extend([-size, grid_y, i, size, grid_y, i])
            colors.extend([0.3]*6)
        
        debug_data = []
        for i in range(0, len(lines), 3):
            debug_data.extend(lines[i:i+3])
            debug_data.extend(colors[i:i+3])
        debug_data = np.array(debug_data, dtype=np.float32)

        self.debug_count = len(lines) // 3
        self.vao_debug = glGenVertexArrays(1); glBindVertexArray(self.vao_debug)
        vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, debug_data.nbytes, debug_data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

        # Axis Gizmo
        gizmo = [0,0,0, 1,0,0, 10,0,0, 1,0,0, 0,0,0, 0,1,0, 0,10,0, 0,1,0, 0,0,0, 0,0,1, 0,0,10, 0,0,1]
        g_data = np.array(gizmo, dtype=np.float32)
        self.vao_gizmo = glGenVertexArrays(1); glBindVertexArray(self.vao_gizmo)
        vbo_g = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo_g)
        glBufferData(GL_ARRAY_BUFFER, g_data.nbytes, g_data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    def _draw_grid(self):
        glUseProgram(self.prog_debug)
        glUniformMatrix4fv(self.loc_d_proj, 1, GL_TRUE, self.proj)
        glUniformMatrix4fv(self.loc_d_view, 1, GL_TRUE, self.view)
        glUniformMatrix4fv(self.loc_d_model, 1, GL_TRUE, np.eye(4, dtype=np.float32))
        if self.vao_debug:
            glBindVertexArray(self.vao_debug)
            glDrawArrays(GL_LINES, 0, self.debug_count)

    def _create_cylinder_mesh(self):
        """Generates a simple unit cylinder (height=1, radius=1) along Y axis."""
        # Simple 12-sided cylinder
        segments = 12
        verts = []; faces = []
        for y in [0.0, 1.0]:
            for i in range(segments):
                theta = 2.0 * np.pi * i / segments
                verts.extend([np.cos(theta), y, np.sin(theta), np.cos(theta), 0, np.sin(theta), i/segments, y])
        for i in range(segments):
            next_i = (i + 1) % segments
            b1, b2, t1, t2 = i, next_i, i+segments, next_i+segments
            faces.extend([b1, t1, b2, b2, t1, t2])
        return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.uint32)

    def init_occluder_primitive(self):
        """Creates the VBO for the generic cylinder occluder."""
        self.makeCurrent()
        v_data, i_data = self._create_cylinder_mesh()
        self.vao_cylinder = glGenVertexArrays(1); glBindVertexArray(self.vao_cylinder)
        vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, v_data.nbytes, v_data, GL_STATIC_DRAW)
        ebo = glGenBuffers(1); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, i_data.nbytes, i_data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        self.idx_count_cylinder = len(i_data)
        self.cylinder_ready = True

    def resizeGL(self, w, h):
        """Calculates a centered viewport that maintains aspect ratio."""
        if h == 0: h = 1 # Prevent divide by zero
        win_aspect = w / h
        
        # Calculate new viewport (Black bars logic)
        if win_aspect > self.img_aspect:
            # Window is too wide (black bars on sides)
            new_h = h; new_w = int(h * self.img_aspect); x_off = (w - new_w) // 2; y_off = 0
        else:
            # Window is too tall (black bars on top/bottom)
            new_w = w; new_h = int(w / self.img_aspect); x_off = 0; y_off = (h - new_h) // 2
        self.viewport_rect = (x_off, y_off, new_w, new_h)

    def update_projection(self, aspect_ratio=None):
        if aspect_ratio is None: aspect_ratio = self.img_aspect
        f = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        zn, zf = self.near_plane, self.far_plane
        self.proj = np.array([[f/aspect_ratio,0,0,0], [0,f,0,0], [0,0,(zf+zn)/(zn-zf), (2*zf*zn)/(zn-zf)], [0,0,-1,0]], dtype=np.float32)

    def paintGL(self):
        glViewport(0, 0, self.width(), self.height())
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        vx, vy, vw, vh = self.viewport_rect
        glViewport(vx, vy, vw, vh)
        self.update_projection(aspect_ratio=self.img_aspect)
        
        # 1. Background
        if self.camera_ready:
            glDisable(GL_DEPTH_TEST); glUseProgram(self.prog_bg)
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, self.cam_tex_id)
            glBindVertexArray(self.vao_bg); glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        
        glEnable(GL_DEPTH_TEST) 
        
        # 2. Debug Grid
        if self.show_grid:
            self._draw_grid()

        # 3. Draw Occluders (Meshes or Procedural)
        if self.occluder_instances:
            glUseProgram(self.prog_occ)
            glUniformMatrix4fv(glGetUniformLocation(self.prog_occ, "u_proj"), 1, GL_TRUE, self.proj)
            glUniformMatrix4fv(glGetUniformLocation(self.prog_occ, "u_view"), 1, GL_TRUE, self.view)

            if self.debug_occluder:
                # RED color for debug
                glUniform4f(glGetUniformLocation(self.prog_occ, "u_color"), 1.0, 0.0, 0.0, 0.5)
            else:
                # Invisible mask
                glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)

            mesh_instances = []
            cyl_instances = []  # cylinders are for wrist , ring, waist .
            for inst in self.occluder_instances:
                if isinstance(inst, dict) and inst.get('mesh_key'):
                    mesh_instances.append(inst)
                else:
                    cyl_instances.append(inst if not isinstance(inst, dict) else inst.get('matrix'))

            # A. Draw Meshes (Head/Body)
            for inst in mesh_instances:
                key = inst.get('mesh_key')
                mesh_data = self.occluder_meshes.get(key)
                
                if not mesh_data:
                    # Optional: Print warning only once to avoid spam
                    log.warning(f"[Render] Warning: Request to draw {key} but mesh not loaded!") 
                    continue
                glBindVertexArray(mesh_data['vao'])
                glUniformMatrix4fv(glGetUniformLocation(self.prog_occ, "u_model"), 1, GL_TRUE, inst['matrix'])
                glDrawElements(GL_TRIANGLES, mesh_data['count'], GL_UNSIGNED_INT, None)

            # B. Draw Procedural Cylinders (Wrist/Ring/Waist)
            if self.cylinder_ready and cyl_instances:
                glBindVertexArray(self.vao_cylinder)
                for instance_mat in cyl_instances:
                    if instance_mat is None:
                        continue
                    glUniformMatrix4fv(glGetUniformLocation(self.prog_occ, "u_model"), 1, GL_TRUE, instance_mat)
                    glDrawElements(GL_TRIANGLES, self.idx_count_cylinder, GL_UNSIGNED_INT, None)

            # Restore Color Mask if it was disabled
            if not self.debug_occluder:
                glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

        # 4. Draw Jewelry 
        if self.meshes and self.render_instances:
            glUseProgram(self.prog_mesh)
            glUniformMatrix4fv(self.loc_m_proj, 1, GL_TRUE, self.proj)
            glUniformMatrix4fv(self.loc_m_view, 1, GL_TRUE, self.view)
            glUniform3f(self.loc_l_pos, *self.light_pos)
            glUniform1f(self.loc_l_amb, self.ambient_str)
            glUniform1f(self.loc_l_diff, self.diffuse_str)
            glUniform1f(glGetUniformLocation(self.prog_mesh, "u_exposure"), self.exposure)
            glUniform1f(glGetUniformLocation(self.prog_mesh, "u_gamma"), self.gamma)   

            for inst in self.render_instances:
                if isinstance(inst, dict):
                    mat = inst['matrix']
                    key = inst.get('model_key', 'default')
                else:
                    mat = inst
                    key = 'default'

                if key in self.meshes:
                    mesh_data = self.meshes[key]
                    glBindVertexArray(mesh_data['vao'])
                    if mesh_data['tex']:
                        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, mesh_data['tex'])
                        glUniform1i(self.loc_m_has_tex, 1)
                    else:
                        glUniform1i(self.loc_m_has_tex, 0)
                        glUniform4f(self.loc_m_color, 1.0, 0.84, 0.0, 1.0)
                    
                    glUniformMatrix4fv(self.loc_m_model, 1, GL_TRUE, mat)
                    glDrawElements(GL_TRIANGLES, mesh_data['count'], GL_UNSIGNED_INT, None)

    def load_object(self, path, key='default', is_occluder=False):
        """Loads a mesh and stores it in the dictionary."""
        # 1. Load Data
        data = load_mesh_data(path, is_occluder)
        if not data: return

        self.makeCurrent()
        
        # 2. Upload to GPU
        interleaved = np.hstack((data.vertices, data.norms, data.uvs)).astype(np.float32)
        
        # Generate VAO/VBO
        vao = glGenVertexArrays(1); glBindVertexArray(vao)
        vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
        ebo = glGenBuffers(1); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.faces.nbytes, data.faces, GL_STATIC_DRAW)
        
        # Layouts
        stride = 32 # 3+3+2 floats * 4 bytes
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        
        # Texture
        tex_id = None
        if data.texture_data is not None:
            tex_id = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, data.texture_data.shape[1], data.texture_data.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, data.texture_data)

        # 3. Store
        self.meshes[key] = {
            'vao': vao,
            'tex': tex_id,
            'count': len(data.faces)
        }

        self.doneCurrent()
        self.update()

    def load_occluder(self, path, key):
        """Loads an occluder mesh and stores it separately."""
        log.info(f"[Renderer] Attempting to load occluder: {key} from {path}")
        
        if key in self.occluder_meshes:
            log.info(f"[Renderer] Occluder {key} already loaded.")
            return

        # 1. Load Data
        data = load_mesh_data(path, is_occluder=True)
        if not data:
            log.error(f"[Renderer] ❌ FAILED to load occluder: {path}")
            return # <--- This is where it likely fails right now

        log.info(f"[Renderer] ✔ Loaded {key}: {len(data.faces)} faces.")

        self.makeCurrent()
        interleaved = np.hstack((data.vertices, data.norms, data.uvs)).astype(np.float32)

        vao = glGenVertexArrays(1); glBindVertexArray(vao)
        vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
        ebo = glGenBuffers(1); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.faces.nbytes, data.faces, GL_STATIC_DRAW)

        stride = 32
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))

        self.occluder_meshes[key] = {
            'vao': vao,
            'count': len(data.faces)
        }

        self.doneCurrent()
        self.update()

    def update_bg(self, frame):
        if frame is None: return
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = f.shape
        if self.cam_w != w or self.cam_h != h:
            self.cam_w, self.cam_h = w, h
            self.img_aspect = w / h
            self.resizeGL(self.width(), self.height())
        self.makeCurrent()
        if not self.cam_tex_id: self.cam_tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.cam_tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, f)
        self.camera_ready = True; self.doneCurrent(); self.update()

    def load_quad(self, image_path, key='default'):
        """
        Loads a 2D image and maps it onto a mathematically generated flat Quad.
        """
        if not image_path or not os.path.exists(image_path):
            log.warning(f"[Renderer] 2D Image not found: {image_path}")
            return

        # 1. Read Image & Calculate Aspect Ratio
        # IMREAD_UNCHANGED ensures we keep the Alpha (transparency) channel!
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            log.error(f"[Renderer] Failed to decode 2D Image: {image_path}")
            return

        h, w = img.shape[:2]
        aspect_ratio = w / float(h)

        # 2. Generate Flat Mesh
        data = create_2d_quad(aspect_ratio)

        self.makeCurrent()

        # 3. Upload Mesh to GPU
        # Flatten the arrays for OpenGL
        interleaved = np.hstack((data.vertices, data.norms, data.uvs)).astype(np.float32)
        flat_faces = data.faces.flatten()

        vao = glGenVertexArrays(1); glBindVertexArray(vao)
        vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
        
        ebo = glGenBuffers(1); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, flat_faces.nbytes, flat_faces, GL_STATIC_DRAW)

        stride = 32 # 3+3+2 floats * 4 bytes
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))

        # 4. Upload Texture to GPU
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Handle color conversions (OpenCV loads as BGR/BGRA, OpenGL needs RGBA)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_gl = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            # If the user uploaded a JPG without transparency, force an alpha channel
            img_gl = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        # OpenGL expects the Y-axis to start at the bottom, so we flip vertically
        img_gl = cv2.flip(img_gl, 0)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_gl)

        # 5. Store in dictionary
        self.meshes[key] = {
            'vao': vao,
            'tex': tex_id,
            'count': len(flat_faces) # Should be 6 indices (2 triangles)
        }

        self.doneCurrent()
        self.update()
        log.info(f"[Renderer] Loaded 2D Sprite '{key}' ({w}x{h})")