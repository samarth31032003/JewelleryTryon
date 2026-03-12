# graphics/overlay_2d.py
import cv2
import numpy as np
import os
from utils.logger import logger

log = logger.bind(component="graphics")

class BaseOverlay2D:
    """Base class handling image loading, resizing, and alpha blending for all 2D assets."""
    def __init__(self, image_path=None):
        self.original_image = None
        self.cached_image = None
        self.cached_size = (0, 0)
        
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.brightness = 1.0
        
        if image_path and os.path.exists(image_path):
            self.load_image(image_path)

    def load_image(self, path):
        if not os.path.exists(path): return False
        
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: return False
        
        # Ensure image has an alpha channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            b, g, r = cv2.split(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Make pure white backgrounds transparent
            _, alpha = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            img = cv2.merge([b, g, r, alpha])
            
        self.original_image = img
        self.cached_image = None
        self.cached_size = (0, 0)
        return True

    def _get_resized(self, target_size):
        if self.original_image is None: return None
        if self.cached_size != target_size:
            self.cached_image = cv2.resize(self.original_image, target_size, interpolation=cv2.INTER_AREA)
            self.cached_size = target_size
        return self.cached_image.copy()

    def _apply_brightness(self, overlay):
        if self.brightness == 1.0: return overlay
        alpha = overlay[:, :, 3] if overlay.shape[2] == 4 else None
        bgr = overlay[:, :, :3]
        bgr = cv2.convertScaleAbs(bgr, alpha=self.brightness, beta=0)
        if alpha is not None:
            return cv2.merge((bgr[:,:,0], bgr[:,:,1], bgr[:,:,2], alpha))
        return bgr

    def _alpha_blend(self, background, overlay, x, y):
        bg_h, bg_w = background.shape[:2]
        ol_h, ol_w = overlay.shape[:2]
        
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bg_w, x + ol_w), min(bg_h, y + ol_h)
        
        ol_x1, ol_y1 = x1 - x, y1 - y
        ol_x2, ol_y2 = ol_x1 + (x2 - x1), ol_y1 + (y2 - y1)
        
        if x2 <= x1 or y2 <= y1: return background
            
        bg_region = background[y1:y2, x1:x2]
        ol_region = overlay[ol_y1:ol_y2, ol_x1:ol_x2]
        
        if ol_region.shape[2] == 4:
            alpha = ol_region[:, :, 3:4].astype(np.float32) / 255.0
            ol_rgb = ol_region[:, :, :3]
        else:
            alpha = np.ones((ol_region.shape[0], ol_region.shape[1], 1), dtype=np.float32)
            ol_rgb = ol_region
            
        blended = (1 - alpha) * bg_region + alpha * ol_rgb
        background[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        return background

    def prepare_and_blend(self, frame, center_x, anchor_y, base_size, head_tilt=0, side_flip=False, y_anchor='center'):
        """
        Unified 2D pipeline: handles resize relative to aspect, rotation, brightness, offset application, and alpha blending.
        """
        if self.original_image is None: return frame
        
        oh, ow = self.original_image.shape[:2]
        aspect = ow / oh if oh > 0 else 1
        new_w, new_h = int(base_size * aspect), int(base_size)
        
        overlay = self._get_resized((new_w, new_h))
        if overlay is None: return frame
            
        if side_flip:
            overlay = cv2.flip(overlay, 1)
            
        if abs(head_tilt) > 2:
            center = (new_w // 2, new_h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, -head_tilt, 1.0)
            overlay = cv2.warpAffine(overlay, rot_mat, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            
        overlay = self._apply_brightness(overlay)
        
        final_x = int(center_x - new_w // 2 + self.offset_x)
        if y_anchor == 'top':
            final_y = int(anchor_y + self.offset_y)
        elif y_anchor == 'necklace':
            final_y = int(anchor_y - new_h // 3 + self.offset_y)
        else: # center
            final_y = int(anchor_y - new_h // 2 + self.offset_y)
            
        return self._alpha_blend(frame, overlay, final_x, final_y)


class ForeheadPendantOverlay(BaseOverlay2D):
    def overlay_on_frame(self, frame, forehead_center, face_width=None, head_tilt=0):
        if self.original_image is None or forehead_center is None: return frame
            
        base_size = int(face_width * 0.25 * self.scale) if face_width else int(80 * self.scale)
        base_size = max(30, min(400, base_size))
        
        return self.prepare_and_blend(frame, forehead_center[0], forehead_center[1], base_size, head_tilt=head_tilt, y_anchor='center')


class EarringOverlay(BaseOverlay2D):
    def __init__(self, image_path=None):
        super().__init__(image_path)
        self.offset_y = 10
        self.mirror_right = True
        
        # Visibility Flags
        self.hide_left = False
        self.hide_right = False

    def overlay_on_frame(self, frame, left_ear, right_ear, face_width=None):
        if self.original_image is None: return frame
            
        base_size = int(face_width * 0.25 * self.scale) if face_width else int(60 * self.scale)
        base_size = max(20, min(300, base_size))
        
        # Fetch the flags dynamically updated by Strategy2D
        hide_left = getattr(self, 'hide_left', False)
        hide_right = getattr(self, 'hide_right', False)
        
        # Draw Left Earring ONLY if it is not hidden
        if left_ear and not hide_left:
            frame = self.prepare_and_blend(frame, left_ear[0], left_ear[1], base_size, y_anchor='top')
                
        # Draw Right Earring ONLY if it is not hidden
        if right_ear and not hide_right:
            old_ox = self.offset_x
            self.offset_x = -old_ox # Invert X offset for right ear
            frame = self.prepare_and_blend(frame, right_ear[0], right_ear[1], base_size, side_flip=self.mirror_right, y_anchor='top')
            self.offset_x = old_ox
                
        return frame


class NosePinOverlay(BaseOverlay2D):
    def __init__(self, image_path=None):
        super().__init__(image_path)
        # Your existing code already had this side tracking!
        self.side = "left" 
        
        # Visibility Flag
        self.hide = False

    def overlay_on_frame(self, frame, nose_point, face_width=None, head_tilt=0):
        # Check if Strategy2D told us to hide
        if getattr(self, 'hide', False): 
            return frame
            
        if self.original_image is None or nose_point is None: return frame
            
        base_size = int(face_width * 0.08 * self.scale) if face_width else int(25 * self.scale)
        base_size = max(10, min(100, base_size))
        
        return self.prepare_and_blend(frame, nose_point[0], nose_point[1], base_size, head_tilt=head_tilt, side_flip=(self.side=="right"), y_anchor='center')

class NecklaceOverlay(BaseOverlay2D):
    def __init__(self, image_path=None):
        super().__init__(image_path)
        self.enable_occlusion = True
        self.enable_perspective = False
        self.enable_curvature = False
        self.curvature_strength = 0.3
        self.reference_face_width = 150

    def overlay_on_frame(self, frame, left_shoulder, right_shoulder, chin_point=None, neck_points=None, face_width=None, head_pose=None):
        if self.original_image is None: return frame
            
        try:
            h, w = frame.shape[:2]
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            if shoulder_width < 10: return frame
            
            # Decoupled scaling dependency on face_lms when missing
            depth_scale = (face_width / self.reference_face_width) if face_width else (shoulder_width / 300.0)
                
            base_width = int(shoulder_width * 1.2 * self.scale * depth_scale)
            base_width = max(50, min(base_width, w))
            
            orig_h, orig_w = self.original_image.shape[:2]
            aspect = orig_h / orig_w
            base_height = int(base_width * aspect)
            
            necklace = cv2.resize(self.original_image, (base_width, base_height), interpolation=cv2.INTER_LINEAR)
                                  
            if self.enable_perspective:
                necklace = self._apply_perspective(necklace, left_shoulder, right_shoulder, head_pose)
                
            if self.enable_curvature:
                necklace = self._apply_curvature(necklace, left_shoulder, right_shoulder)
                
            shoulder_center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
            shoulder_center_y = int((left_shoulder[1] + right_shoulder[1]) / 2)
            
            nl_h, nl_w = necklace.shape[:2]
            pos_x = shoulder_center_x - nl_w // 2 + self.offset_x
            pos_y = shoulder_center_y - nl_h // 3 + self.offset_y
            
            if self.enable_occlusion and chin_point is not None:
                necklace = self._apply_occlusion(necklace, pos_x, pos_y, chin_point, neck_points, shoulder_center_y)
                
            necklace = self._apply_brightness(necklace)
                                                  
            frame = self._alpha_blend(frame, necklace, pos_x, pos_y)
            
        except Exception as e:
            log.error(f"[Necklace Overlay Error] {e}")
            
        return frame

    def _apply_perspective(self, necklace, left_shoulder, right_shoulder, head_pose=None):
        nl_h, nl_w = necklace.shape[:2]
        dx, dy = right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1]
        shoulder_angle = np.arctan2(dy, dx)
        shoulder_width = np.sqrt(dx**2 + dy**2)
        
        yaw_estimate = head_pose[0] * 0.01 if head_pose is not None else (dy / max(shoulder_width, 1)) * 0.3
            
        src_pts = np.float32([[0, 0], [nl_w, 0], [nl_w, nl_h], [0, nl_h]])
        perspective_amount = yaw_estimate * nl_w * 0.15
        tilt_y = np.sin(shoulder_angle) * nl_h * 0.1
        
        dst_pts = np.float32([
            [perspective_amount, -tilt_y], [nl_w - perspective_amount, tilt_y],
            [nl_w + perspective_amount, nl_h + tilt_y], [-perspective_amount, nl_h - tilt_y]
        ])
        
        min_x, min_y = min(dst_pts[:, 0]), min(dst_pts[:, 1])
        if min_x < 0: dst_pts[:, 0] -= min_x
        if min_y < 0: dst_pts[:, 1] -= min_y
            
        new_w, new_h = int(max(dst_pts[:, 0])) + 1, int(max(dst_pts[:, 1])) + 1
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(necklace, matrix, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        
    def _apply_curvature(self, necklace, left_shoulder, right_shoulder):
        nl_h, nl_w = necklace.shape[:2]
        if nl_h < 10 or nl_w < 10 or self.curvature_strength < 0.01: return necklace
            
        y_coords, x_coords = np.mgrid[0:nl_h, 0:nl_w].astype(np.float32)
        x_norm = (x_coords / (nl_w - 1)) * 2 - 1
        y_norm = y_coords / (nl_h - 1)
        
        curve_factor = self.curvature_strength * (1 - y_norm * 0.5)
        y_disp = curve_factor * (1 - x_norm**2) * nl_h * 0.15
        x_disp = x_norm * y_norm * nl_w * 0.05 * self.curvature_strength
        
        map_x = np.clip(x_coords - x_disp, 0, nl_w - 1)
        map_y = np.clip(y_coords - y_disp, 0, nl_h - 1)
        
        return cv2.remap(necklace, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        
    def _apply_occlusion(self, necklace, pos_x, pos_y, chin_point, neck_points, shoulder_y):
        try:
            nl_h, nl_w = necklace.shape[:2]
            chin_local_x, chin_local_y = chin_point[0] - pos_x, chin_point[1] - pos_y
            
            if 0 < chin_local_x < nl_w and chin_local_y > 0:
                neck_width, center_x = nl_w // 3, nl_w // 2
                ellipse_height = max(20, chin_local_y)
                
                if neck_points:
                    neck_left, neck_right = neck_points[0][0] - pos_x, neck_points[1][0] - pos_x
                    neck_width = max(20, abs(neck_right - neck_left) // 2)
                    
                y_range = min(ellipse_height + 30, nl_h)
                y_coords = np.arange(y_range).reshape(-1, 1)
                x_coords = np.arange(nl_w).reshape(1, -1)
                
                dx = np.abs(x_coords - center_x)
                fade = np.clip(1.0 - (y_coords / (ellipse_height + 30)), 0, 1)
                
                mask_values = np.where(dx < neck_width, 1 - fade * 0.85, 1.0)
                necklace[:y_range, :, 3] = (necklace[:y_range, :, 3] * mask_values).astype(np.uint8)
        except Exception:
            pass
        return necklace


class CollectionManager2D:
    def __init__(self):
        self.overlays = {}
        
    def load_item(self, key, image_path):
        """Loads a 2D image using the strict keys provided by TryOnWindow"""
        if key == "forehead": self.overlays[key] = ForeheadPendantOverlay(image_path)
        elif key == "ear": self.overlays[key] = EarringOverlay(image_path)
        elif key == "necklace": self.overlays[key] = NecklaceOverlay(image_path)
        elif key == "nosepin": self.overlays[key] = NosePinOverlay(image_path)
        else:
            log.warning(f"[2D Manager] Unknown 2D component: {key}")
            return
            
        if self.overlays[key].original_image is not None:
            log.info(f"[2D Manager] Loaded {key}: {image_path}")
            
    def clear(self):
        self.overlays = {}

    def update_settings(self, scale, offset_x, offset_y, brightness):
        for overlay in self.overlays.values():
            overlay.scale = scale
            overlay.offset_x = offset_x
            overlay.offset_y = offset_y
            overlay.brightness = brightness
        
    def process_frame(self, frame, results, w, h):
        """Cleanly unpacks landmarks and routes them to active overlays."""
        if not self.overlays: return frame
        
        composited = frame.copy()
        
        # 1. Clean Landmark Extraction (Using new Tasks API format)
        face_lms = getattr(results.face_landmarks, 'landmark', None) if getattr(results, 'face_landmarks', None) else None
        pose_lms = getattr(results.pose_landmarks, 'landmark', None) if getattr(results, 'pose_landmarks', None) else None
        
        # 2. Pre-calculate shared face metrics if a face exists
        face_width, head_tilt = None, 0
        if face_lms:
            left_ear, right_ear = face_lms[234], face_lms[454]
            face_width = abs(right_ear.x - left_ear.x) * w
            
            eye_dx = (face_lms[263].x - face_lms[33].x) * w
            eye_dy = (face_lms[263].y - face_lms[33].y) * h
            head_tilt = np.degrees(np.arctan2(eye_dy, eye_dx))

        # 3. Process each overlay strictly by key
        if "forehead" in self.overlays and face_lms:
            hairline, glabella = face_lms[151], face_lms[9]
            center = (int(hairline.x * w), int((hairline.y + glabella.y) / 2 * h))
            composited = self.overlays["forehead"].overlay_on_frame(composited, center, face_width, head_tilt)

        if "ear" in self.overlays and face_lms:
            left_pt = (int(face_lms[234].x * w), int(face_lms[234].y * h))
            right_pt = (int(face_lms[454].x * w), int(face_lms[454].y * h))
            composited = self.overlays["ear"].overlay_on_frame(composited, left_pt, right_pt, face_width)

        if "nosepin" in self.overlays and face_lms:
            ov = self.overlays["nosepin"]
            if ov.side == "left":
                nx, ny = (face_lms[129].x + face_lms[219].x) / 2, (face_lms[129].y + face_lms[219].y) / 2
            else:
                nx, ny = (face_lms[358].x + face_lms[439].x) / 2, (face_lms[358].y + face_lms[439].y) / 2
            composited = ov.overlay_on_frame(composited, (int(nx * w), int(ny * h)), face_width, head_tilt)

        if "necklace" in self.overlays and pose_lms:
            # 11 is Left Shoulder, 12 is Right Shoulder
            l_shoulder, r_shoulder = pose_lms[11], pose_lms[12]
            
            if l_shoulder.visibility > 0.5 and r_shoulder.visibility > 0.5:
                ls_pt = (int(l_shoulder.x * w), int(l_shoulder.y * h))
                rs_pt = (int(r_shoulder.x * w), int(r_shoulder.y * h))
                
                chin_pt, neck_pts = None, None
                # Incorporate face metrics for occlusion *if* the face is visible
                if face_lms:
                    chin_pt = (int(face_lms[152].x * w), int(face_lms[152].y * h))
                    mid_x = (ls_pt[0] + rs_pt[0]) // 2
                    neck_y = (ls_pt[1] + chin_pt[1]) // 2
                    neck_w = abs(rs_pt[0] - ls_pt[0]) // 4
                    neck_pts = ((mid_x - neck_w, neck_y), (mid_x + neck_w, neck_y))

                composited = self.overlays["necklace"].overlay_on_frame(
                    composited, ls_pt, rs_pt, chin_pt, neck_pts, face_width, None
                )

        return composited