from __future__ import annotations

import numpy as np


class BallKalmanTracker:
    """
    Kalman filter for ball position in image space.
    State: [x, y, vx, vy] (position + velocity in pixels)
    Measurement: [x, y] (detected position)
    """

    def __init__(self, process_noise: float = 0.1):
        """Initialize Kalman filter matrices."""
        self._dt = 1.0  # time step (1 frame)

        # State vector: [x, y, vx, vy]
        self._state = np.zeros(4, dtype=np.float64)

        # State transition matrix (constant velocity model)
        self._F = np.array(
            [
                [1, 0, self._dt, 0],
                [0, 1, 0, self._dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Measurement matrix (we observe x, y)
        self._H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float64,
        )

        # Process noise covariance
        q = process_noise
        self._Q = np.array(
            [
                [q * 0.25, 0, q * 0.5, 0],
                [0, q * 0.25, 0, q * 0.5],
                [q * 0.5, 0, q, 0],
                [0, q * 0.5, 0, q],
            ],
            dtype=np.float64,
        )

        # Measurement noise covariance
        self._R = np.eye(2, dtype=np.float64) * 5.0

        # Error covariance matrix
        self._P = np.eye(4, dtype=np.float64) * 100.0

        self._initialized = False

    def predict(self) -> tuple[float, float]:
        """Predict next position. Returns (x, y)."""
        if not self._initialized:
            return (self._state[0], self._state[1])

        # Predict state
        self._state = self._F @ self._state

        # Predict covariance
        self._P = self._F @ self._P @ self._F.T + self._Q

        return (float(self._state[0]), float(self._state[1]))

    def update(self, measurement: tuple[float, float], confidence: float = 1.0):
        """Update with detection. Confidence scales measurement noise."""
        z = np.array([measurement[0], measurement[1]], dtype=np.float64)

        if not self._initialized:
            self._state[0] = z[0]
            self._state[1] = z[1]
            self._state[2] = 0.0
            self._state[3] = 0.0
            self._initialized = True
            return

        # Scale measurement noise by inverse confidence (lower confidence = more noise)
        noise_scale = 1.0 / max(confidence, 0.1)
        R = self._R * noise_scale

        # Innovation (measurement residual)
        y = z - self._H @ self._state

        # Innovation covariance
        S = self._H @ self._P @ self._H.T + R

        # Kalman gain
        K = self._P @ self._H.T @ np.linalg.inv(S)

        # Update state
        self._state = self._state + K @ y

        # Update covariance
        eye = np.eye(4, dtype=np.float64)
        self._P = (eye - K @ self._H) @ self._P

    def get_state(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return (position_xy, velocity_xy)."""
        pos = (float(self._state[0]), float(self._state[1]))
        vel = (float(self._state[2]), float(self._state[3]))
        return pos, vel

    @property
    def initialized(self) -> bool:
        """Whether the tracker has received at least one measurement."""
        return self._initialized
