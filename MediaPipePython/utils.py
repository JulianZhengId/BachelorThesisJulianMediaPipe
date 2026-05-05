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

class KalmanFilter1D:
    """
    Kalman filter for a single dimension of a MediaPipe landmark.
    State: [position, velocity, acceleration]
    """

    def __init__(self, process_noise=1e-4, measurement_noise=1e-2, dt=1/30):
        self.dt = dt

        # State: [position, velocity, acceleration]
        self.x = np.zeros(3)

        # State transition matrix
        self.F = np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1,  dt],
            [0, 0,  1]
        ])

        # Measurement matrix (we only observe position)
        self.H = np.array([[1, 0, 0]])

        # Covariance matrix
        self.P = np.eye(3) * 1.0

        # Process noise
        #self.Q = np.eye(3) * process_noise
        self.Q = process_noise * np.array([
            [dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2],
            [dt ** 3 / 2, dt ** 2, dt],
            [dt ** 2 / 2, dt, 1]
        ])

        # Measurement noise
        self.R = np.array([[measurement_noise]])

        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.x[0] = measurement
            self.initialized = True
            return measurement

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = measurement - self.H @ x_pred              # residual
        S = self.H @ P_pred @ self.H.T + self.R        # residual covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)       # kalman gain

        self.x = x_pred + (K @ y).flatten()
        self.P = (np.eye(3) - K @ self.H) @ P_pred

        return self.x[0]


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

    def update(self, landmarks):
        """
        landmarks: array of shape (num_landmarks, num_dims)
                   e.g. (21, 3) for hand
        Returns: filtered landmarks, same shape
        """
        filtered = np.zeros_like(landmarks)
        for i in range(self.num_landmarks):
            for d in range(self.num_dims):
                filtered[i, d] = self.filters[i][d].update(landmarks[i, d])
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
