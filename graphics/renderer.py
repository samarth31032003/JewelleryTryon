# graphics/renderer.py
import ctypes
import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PyQt5.QtWidgets import QOpenGLWidget

from graphics.shaders import *
from utils.mesh_loader import load_mesh_data

class ARViewerWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # State Flags
        self.mesh_ready = False
        self.occluder_ready = False
        self.camera_ready = False
        self.show_grid = False
        self.debug_occluder = True 

        self.vao_debug = None
        self.debug_count = 0
        self.vao_gizmo = None  # Axis Gizmo for the object
        
        # GL IDs
        self.mesh_tex_id = None
        self.cam_tex_id = None
        self.vao_mesh = None
        self.vao_occluder = None
        
        # Matrices
        self.proj = np.eye(4, dtype=np.float32)
        self.view = np.eye(4, dtype=np.float32)
        self.model_bracelet = np.eye(4, dtype=np.float32)
        self.model_occluder = np.eye(4, dtype=np.float32)
        
        # Scene Settings
        self.fov = 40.0
        self.near_plane = 0.1
        self.far_plane = 1000.0
        self.light_pos = [0.0, 10.0, 10.0]
        self.ambient_str = 0.4
        self.diffuse_str = 0.8
        self.w_w, self.w_h = 800, 600

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
        # Compile Shaders
        self.prog_mesh = compileProgram(compileShader(MESH_VS, GL_VERTEX_SHADER), compileShader(MESH_FS, GL_FRAGMENT_SHADER))
        self.prog_bg = compileProgram(compileShader(BG_VS, GL_VERTEX_SHADER), compileShader(BG_FS, GL_FRAGMENT_SHADER))
        # Compile New Debug Shader
        self.prog_debug = compileProgram(compileShader(DEBUG_VS, GL_VERTEX_SHADER), compileShader(DEBUG_FS, GL_FRAGMENT_SHADER))
       
        # Cache Uniform Locations (Mesh)
        self.loc_m_model = glGetUniformLocation(self.prog_mesh, "u_model")
        self.loc_m_view = glGetUniformLocation(self.prog_mesh, "u_view")
        self.loc_m_proj = glGetUniformLocation(self.prog_mesh, "u_proj")
        self.loc_m_tex = glGetUniformLocation(self.prog_mesh, "u_tex")
        self.loc_m_has_tex = glGetUniformLocation(self.prog_mesh, "u_has_tex")
        self.loc_m_color = glGetUniformLocation(self.prog_mesh, "u_color_override")
        self.loc_l_pos = glGetUniformLocation(self.prog_mesh, "u_light_pos")
        self.loc_l_amb = glGetUniformLocation(self.prog_mesh, "u_ambient_str")
        self.loc_l_diff = glGetUniformLocation(self.prog_mesh, "u_diffuse_str")
        
        # Cache Uniform Locations (Grid)
        self.loc_d_model = glGetUniformLocation(self.prog_debug, "u_model")
        self.loc_d_view = glGetUniformLocation(self.prog_debug, "u_view")
        self.loc_d_proj = glGetUniformLocation(self.prog_debug, "u_proj")
        
        self._init_bg_quad()
        self._init_debug_layer() 

    def _init_bg_quad(self):
        data = np.array([-1,-1,0,1, 1,-1,1,1, -1,1,0,0, 1,1,1,0], dtype=np.float32)
        self.vao_bg = glGenVertexArrays(1); glBindVertexArray(self.vao_bg)
        vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

    def _init_debug_layer(self):
        # 1. FLOOR GRID (Gray Lines) - Static World Grid
        lines = []
        colors = []
        size = 50  # Huge grid
        step = 5
        grid_y = -10 # Push grid down so it doesn't cut through the bracelet
        
        # Create lines for a floor grid
        for i in range(-size, size + 1, step):
            # Lines along X
            lines.extend([i, grid_y, -size,  i, grid_y, size])
            colors.extend([0.3, 0.3, 0.3,    0.3, 0.3, 0.3]) # Gray
            # Lines along Z
            lines.extend([-size, grid_y, i,  size, grid_y, i])
            colors.extend([0.3, 0.3, 0.3,    0.3, 0.3, 0.3])

        self.debug_count = len(lines) // 3
        data_grid = np.array(lines + colors, dtype=np.float32).reshape(-1) # Flatten logic
        
        # Interleaved: [x,y,z, r,g,b]... actually simpler to do separated for lines but let's stack
        # Correct packing: Vertex 1 [Pos, Col], Vertex 2 [Pos, Col]...
        # Let's rebuild properly:
        debug_data = []
        for i in range(0, len(lines), 3):
            # Pos (x,y,z)
            debug_data.extend(lines[i:i+3])
            # Col (r,g,b) -> corresponding index in colors
            debug_data.extend(colors[i:i+3])
            
        debug_data = np.array(debug_data, dtype=np.float32)

        self.vao_debug = glGenVertexArrays(1); glBindVertexArray(self.vao_debug)
        vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, debug_data.nbytes, debug_data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

        # 2. AXIS GIZMO (Red/Green/Blue) - Attached to Object
        # Line starts at 0,0,0 and goes to 10,0,0
        gizmo_verts = [
            0,0,0,  1,0,0,  # X Axis (Red)
            10,0,0, 1,0,0,
            0,0,0,  0,1,0,  # Y Axis (Green)
            0,10,0, 0,1,0,
            0,0,0,  0,0,1,  # Z Axis (Blue)
            0,0,10, 0,0,1
        ]
        g_data = np.array(gizmo_verts, dtype=np.float32)
        
        self.vao_gizmo = glGenVertexArrays(1); glBindVertexArray(self.vao_gizmo)
        vbo_g = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo_g)
        glBufferData(GL_ARRAY_BUFFER, g_data.nbytes, g_data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        self.w_w, self.w_h = w, h
        self.update_projection()

    def update_projection(self):
        aspect = self.w_w / self.w_h if self.w_h > 0 else 1.0
        f = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        zn, zf = self.near_plane, self.far_plane
        self.proj = np.array([[f/aspect,0,0,0], [0,f,0,0], [0,0,(zf+zn)/(zn-zf),(2*zf*zn)/(zn-zf)], [0,0,-1,0]], dtype=np.float32)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 1. Draw Background
        if self.camera_ready:
            glDisable(GL_DEPTH_TEST); glUseProgram(self.prog_bg)
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, self.cam_tex_id)
            glBindVertexArray(self.vao_bg); glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        
        glEnable(GL_DEPTH_TEST) 
        
        # 2. DRAW DEBUG LAYER ---
        if self.show_grid:
            glUseProgram(self.prog_debug)
            glUniformMatrix4fv(self.loc_d_proj, 1, GL_TRUE, self.proj)
            glUniformMatrix4fv(self.loc_d_view, 1, GL_TRUE, self.view)
            
            # 1. Draw Floor Grid (Identity Model)
            identity = np.eye(4, dtype=np.float32)
            glUniformMatrix4fv(self.loc_d_model, 1, GL_TRUE, identity)
            glBindVertexArray(self.vao_debug)
            glDrawArrays(GL_LINES, 0, self.debug_count)
            
            # 2. Draw Axis Gizmo (Attached to Bracelet!)
            # We use the bracelet's model matrix so the axes rotate WITH the bracelet
            if self.mesh_ready:
                # We disable depth test so you can see axis even if inside the object
                glDisable(GL_DEPTH_TEST) 
                glLineWidth(3.0) # Make axes thick
                glUniformMatrix4fv(self.loc_d_model, 1, GL_TRUE, self.model_bracelet)
                glBindVertexArray(self.vao_gizmo)
                glDrawArrays(GL_LINES, 0, 6) # 3 lines * 2 verts
                glLineWidth(1.0)
                glEnable(GL_DEPTH_TEST)
        
        # 3. Draw Meshes
        glUseProgram(self.prog_mesh)
        glUniformMatrix4fv(self.loc_m_proj, 1, GL_TRUE, self.proj)
        glUniformMatrix4fv(self.loc_m_view, 1, GL_TRUE, self.view)
        glUniform3f(self.loc_l_pos, self.light_pos[0], self.light_pos[1], self.light_pos[2])
        glUniform1f(self.loc_l_amb, self.ambient_str)
        glUniform1f(self.loc_l_diff, self.diffuse_str)
        
        # Occluder
        if self.occluder_ready:
            glUniformMatrix4fv(self.loc_m_model, 1, GL_TRUE, self.model_occluder)
            glUniform1i(self.loc_m_has_tex, 0) 
            if self.debug_occluder:
                glUniform4f(self.loc_m_color, 1.0, 0.0, 0.0, 0.5) 
                glBindVertexArray(self.vao_occluder)
                glDrawElements(GL_TRIANGLES, self.index_count_occluder, GL_UNSIGNED_INT, None)
            else:
                glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE) 
                glDepthMask(GL_TRUE)                                
                glBindVertexArray(self.vao_occluder)
                glDrawElements(GL_TRIANGLES, self.index_count_occluder, GL_UNSIGNED_INT, None)
                glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)     
        
        # Bracelet
        if self.mesh_ready:
            glUniformMatrix4fv(self.loc_m_model, 1, GL_TRUE, self.model_bracelet)
            glUniform4f(self.loc_m_color, 1.0, 0.84, 0.0, 1.0) 
            if self.mesh_tex_id:
                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, self.mesh_tex_id)
                glUniform1i(self.loc_m_tex, 0); glUniform1i(self.loc_m_has_tex, 1)
            else: glUniform1i(self.loc_m_has_tex, 0)
            glBindVertexArray(self.vao_mesh); glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)

    def load_object(self, path, is_occluder=False):
        """Uses the Utility loader to get data, then uploads to GPU."""
        data = load_mesh_data(path, is_occluder)
        if not data: return

        self.makeCurrent()
        
        # Flatten Data for VBO [Verts, Norms, UVs]
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

        # Store IDs based on type
        if is_occluder:
            self.vao_occluder = vao
            self.index_count_occluder = len(data.faces)
            self.occluder_ready = True
        else:
            self.vao_mesh = vao
            self.mesh_tex_id = tex_id
            self.index_count = len(data.faces)
            self.mesh_ready = True

        self.doneCurrent()
        self.update()

    def update_bg(self, frame):
        if frame is None: return
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = f.shape
        self.makeCurrent()
        if not self.cam_tex_id: self.cam_tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.cam_tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, f)
        self.camera_ready = True; self.doneCurrent(); self.update()