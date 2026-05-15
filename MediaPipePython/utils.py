from mediapipe.tasks.python.components.containers.landmark import Landmark
import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd

class ButterworthFilter:
    def __init__(self, cutoff: float, sampling_rate: float, order: int = 4):
        self.cutoff = cutoff
        self.sampling_rate = sampling_rate
        self.order = order

        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff / nyquist
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return filtfilt(self.b, self.a, data)


class OneEuroFilter:
    def __init__(self, sampling_rate=30.0, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        """
        min_cutoff: minimum cutoff frequency (Hz) — controls smoothing at rest
        beta:       speed coefficient — controls lag during fast motion
        d_cutoff:   cutoff for derivative (usually left at 1.0)
        """
        self.fps = sampling_rate
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        # Smoothing factor derived from cutoff frequency
        # Assumes ~30fps; replace 30 with your actual frame rate
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.fps
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x

        # Estimate derivative
        dx = (x - self.x_prev) * self.fps  # multiply by fps

        # Smooth the derivative
        alpha_d = self._alpha(self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        # Compute speed-adaptive cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # Smooth the signal
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat

import numpy as np

class KalmanFilter1D:
    """
    Kalman filter for a single dimension of a MediaPipe landmark.
    State: [position, velocity, acceleration]
    Uses RTS (Rauch-Tung-Striebel) smoother for zero-phase optimal filtering.
    """

    def __init__(self, process_noise=1e-4, measurement_noise=1e-2, dt=1/30):
        self.dt = dt

        # State transition matrix — constant acceleration model
        self.F = np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1,  dt],
            [0, 0,  1]
        ])

        # Measurement matrix — only position is observed
        self.H = np.array([[1, 0, 0]])

        # Initial covariance
        self.P0 = np.eye(3) * 1.0

        # Process noise covariance — constant acceleration noise model
        self.Q = process_noise * np.array([
            [dt**4/4, dt**3/2, dt**2/2],
            [dt**3/2, dt**2,   dt],
            [dt**2/2, dt,      1]
        ])

        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])

    def smooth(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply RTS smoother to a full 1D signal.
        signal: array of shape (T,) — one coordinate dimension over all frames
        Returns: smoothed signal of shape (T,)
        """
        n = len(signal)

        # Storage
        xs     = np.zeros((n, 3))      # post-update states
        Ps     = np.zeros((n, 3, 3))   # post-update covariances
        Ps_pred = np.zeros((n, 3, 3))  # pre-update predicted covariances

        # --- Initialise at t=0 (update only, no prediction) ---
        x = np.zeros(3)
        x[0] = signal[0]              # initialise position with first measurement
        P = self.P0.copy()

        # Update at t=0 without prediction step
        y = signal[0] - self.H @ x
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        x = x + (K @ y).flatten()
        P = (np.eye(3) - K @ self.H) @ P

        xs[0]      = x
        Ps[0]      = P
        Ps_pred[0] = P  # no prediction at t=0, use post-update as placeholder

        # --- Forward pass t=1 to n-1 ---
        for t in range(1, n):
            # Predict
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q
            Ps_pred[t] = P_pred        # store pre-update covariance

            # Update
            y = signal[t] - self.H @ x_pred
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            x = x_pred + (K @ y).flatten()
            P = (np.eye(3) - K @ self.H) @ P_pred

            xs[t] = x                  # store post-update state
            Ps[t] = P                  # store post-update covariance

        # --- Backward pass (RTS smoother) ---
        xs_smooth = xs.copy()
        Ps_smooth = Ps.copy()

        for t in range(n - 2, -1, -1):
            # Smoother gain: G_t = P_t @ F.T @ inv(P_{t+1|t})
            G = Ps[t] @ self.F.T @ np.linalg.inv(Ps_pred[t + 1])

            # Smoothed state
            xs_smooth[t] = xs[t] + G @ (xs_smooth[t + 1] - self.F @ xs[t])

            # Smoothed covariance
            Ps_smooth[t] = Ps[t] + G @ (Ps_smooth[t + 1] - Ps_pred[t + 1]) @ G.T

        return xs_smooth[:, 0]  # return position only


class KalmanFilterND:
    """
    Manages one KalmanFilter1D per dimension per landmark.
    Works with MediaPipe hand landmarks (21 landmarks, x/y/z each).
    """

    def __init__(self, num_landmarks=21, num_dims=3,
                 process_noise=1e-4, measurement_noise=1e-2, fps=30):
        self.filters = [
            [KalmanFilter1D(process_noise, measurement_noise, 1/fps)
             for _ in range(num_dims)]
            for _ in range(num_landmarks)
        ]
        self.num_landmarks = num_landmarks
        self.num_dims = num_dims

    def smooth(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply RTS smoother to all landmarks.
        landmarks: array of shape (num_frames, num_landmarks, num_dims)
        Returns: smoothed landmarks, same shape
        """
        filtered = np.zeros_like(landmarks)
        for i in range(self.num_landmarks):
            for d in range(self.num_dims):
                signal = landmarks[:, i, d]       # shape (num_frames,)
                filtered[:, i, d] = self.filters[i][d].smooth(signal)
        return filtered

class LandmarksProcessor:

    @staticmethod
    def parse_keypoint_3d(keypoint_3d: list[Landmark]) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d[i].x
            keypoint[i][1] = keypoint_3d[i].y
            keypoint[i][2] = keypoint_3d[i].z
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param keypoint_3d_array: keypoints detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in normal convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]
        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]
        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)
        normal = v[2, :]
        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)
        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

class FilesManager:
    @staticmethod
    def save_landmarks_data(landmarks_data:np.array, columns, filepath):
        df = pd.DataFrame(landmarks_data, columns=columns)
        df.to_csv(filepath, index=False)

    @staticmethod
    def load_landmarks_data(path):
        df = pd.read_csv(path)
        data = df.to_numpy()
        return data.reshape(-1, 21, 3)
