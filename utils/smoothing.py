# utils/smoothing.py
import numpy as np
import cv2

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def update(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            self.dx_prev = np.zeros_like(x)
            return x

        dt = t - self.t_prev
        if dt <= 0: return self.x_prev

        dx = (x - self.x_prev) / dt
        dx_hat = self._exponential_smoothing(dx, self.dx_prev, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = self._exponential_smoothing(x, self.x_prev, alpha)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def _exponential_smoothing(self, current, prev, alpha):
        return alpha * current + (1.0 - alpha) * prev

class RotationFilter:
    """
    Wraps OneEuroFilter to safely smooth rotations using Quaternions.
    Prevents the 'Fidget Spinner' effect (Wrap-around).
    """
    def __init__(self, min_cutoff=1.0, beta=0.0):
        # We filter 4 values (x,y,z,w) of the quaternion
        self.filter = OneEuroFilter(min_cutoff, beta)
    
    def update(self, rvec, t):
        # 1. Convert rvec (Axis-Angle) to Quaternion
        # Angle is length of rvec
        theta = np.linalg.norm(rvec)
        if theta < 1e-6:
            q_curr = np.array([0., 0., 0., 1.])
        else:
            axis = rvec / theta
            s = np.sin(theta / 2)
            c = np.cos(theta / 2)
            # OpenCV convention: (x, y, z, w)
            q_curr = np.array([axis[0]*s, axis[1]*s, axis[2]*s, c]).flatten()

        # 2. Check for Flip (The "Shortest Path" Fix)
        # If the dot product with the previous quaternion is negative,
        # we are on the "other side" of the sphere (179 vs -179 issue).
        # We negate the new quaternion to force the "short way" around.
        if self.filter.x_prev is not None:
            prev_q = self.filter.x_prev
            if np.dot(q_curr, prev_q) < 0:
                q_curr = -q_curr

        # 3. Smooth the Quaternion components
        q_smooth = self.filter.update(q_curr, t)

        # 4. Normalize (Essential for valid rotation)
        q_smooth /= np.linalg.norm(q_smooth)

        # 5. Convert back to rvec
        # acos of w gives the angle
        # Safety clamp to avoid NaN
        w = np.clip(q_smooth[3], -1.0, 1.0)
        theta_new = 2 * np.arccos(w)
        s_new = np.sin(theta_new / 2)
        
        if s_new < 1e-6:
            return np.zeros((3, 1))
        
        axis_new = q_smooth[:3] / s_new
        rvec_smooth = axis_new * theta_new
        
        return rvec_smooth.reshape(3, 1)