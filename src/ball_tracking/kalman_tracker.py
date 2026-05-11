from __future__ import annotations

import numpy as np


class BallKalmanTracker:
    """
    Kalman filter for ball position in image space.
    State: [x, y, vx, vy, ax, ay] (position + velocity + acceleration in pixels)
    Measurement: [x, y] (detected position)
    """

    def __init__(
        self,
        process_noise: float = 0.1,
        gate_threshold: float = 0.0,
        min_pixel_radius: float = 0.0,
        q_gap_factor: float = 0.0,
        q_speed_factor: float = 0.0,
    ):
        dt = 1.0
        self._dt = dt
        dt2 = dt * dt

        self._state = np.zeros(6, dtype=np.float64)

        self._F = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt2, 0],
                [0, 1, 0, dt, 0, 0.5 * dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        self._H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )

        # Discrete-noise model for constant-acceleration filter.
        # Jerk as noise source: G = [dt²/2, dt, 1]^T, Q_1d = G @ G^T * q.
        # Block-diagonal for independent X and Y axes.
        q = process_noise
        g = np.array([dt2 / 2, dt, 1.0])
        q_block = np.outer(g, g) * q
        self._Q_base = np.zeros((6, 6), dtype=np.float64)
        self._Q_base[0:5:2, 0:5:2] = q_block  # x, vx, ax
        self._Q_base[1:6:2, 1:6:2] = q_block  # y, vy, ay

        self._R = np.eye(2, dtype=np.float64) * 5.0
        self._P = np.eye(6, dtype=np.float64) * 100.0

        self._gate_threshold = gate_threshold
        self._min_pixel_radius = min_pixel_radius
        self._q_gap_factor = q_gap_factor
        self._q_speed_factor = q_speed_factor

        self._initialized = False

    def predict(self, gap_len: int = 0) -> tuple[float, float]:
        """Predict next position. Returns (x, y)."""
        if not self._initialized:
            return (self._state[0], self._state[1])

        speed = float(np.sqrt(self._state[2] ** 2 + self._state[3] ** 2))
        q_scale = 1.0 + self._q_gap_factor * gap_len + self._q_speed_factor * speed
        Q_eff = self._Q_base * q_scale

        self._state = self._F @ self._state
        self._P = self._F @ self._P @ self._F.T + Q_eff

        return (float(self._state[0]), float(self._state[1]))

    def update(self, measurement: tuple[float, float], confidence: float = 1.0) -> bool:
        """Update with detection. Returns False if gated (rejected)."""
        z = np.array([measurement[0], measurement[1]], dtype=np.float64)

        if not self._initialized:
            self._state[0] = z[0]
            self._state[1] = z[1]
            self._state[2] = 0.0
            self._state[3] = 0.0
            self._state[4] = 0.0
            self._state[5] = 0.0
            self._initialized = True
            return True

        # Scale measurement noise by inverse confidence (lower confidence = more noise)
        noise_scale = 1.0 / max(confidence, 0.1)
        R = self._R * noise_scale

        # Innovation (measurement residual)
        y = z - self._H @ self._state

        # Innovation covariance
        S = self._H @ self._P @ self._H.T + R

        # Gated innovation: reject if Mahalanobis distance exceeds threshold
        # AND pixel distance exceeds min_pixel_radius (avoids rejecting
        # physically close detections when P is overly tight).
        if self._gate_threshold > 0.0:
            S_inv = np.linalg.inv(S)
            mahal_sq = float(y @ S_inv @ y)
            pixel_dist = float(np.sqrt(y[0] ** 2 + y[1] ** 2))
            if mahal_sq > self._gate_threshold ** 2 and pixel_dist > self._min_pixel_radius:
                return False

        # Kalman gain
        K = self._P @ self._H.T @ np.linalg.inv(S)

        # Update state
        self._state = self._state + K @ y

        # Update covariance
        eye = np.eye(6, dtype=np.float64)
        self._P = (eye - K @ self._H) @ self._P
        return True

    def get_state(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return (position_xy, velocity_xy)."""
        pos = (float(self._state[0]), float(self._state[1]))
        vel = (float(self._state[2]), float(self._state[3]))
        return pos, vel

    def get_full_state(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Return (position_xy, velocity_xy, acceleration_xy)."""
        pos = (float(self._state[0]), float(self._state[1]))
        vel = (float(self._state[2]), float(self._state[3]))
        acc = (float(self._state[4]), float(self._state[5]))
        return pos, vel, acc

    @property
    def initialized(self) -> bool:
        """Whether the tracker has received at least one measurement."""
        return self._initialized
