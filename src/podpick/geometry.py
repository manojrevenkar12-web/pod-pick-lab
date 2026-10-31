"""Camera model and small geometry helpers shared by the pick cell and the simulator.

The tower camera looks into the bin along +y. Image u grows with +x, image v grows with -z.
Depth is the distance along the optical axis (y - camera y), in metres; 0 marks no return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    position: tuple[float, float, float]
    hfov_deg: float

    @property
    def f(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def cx(self) -> float:
        return self.width / 2

    @property
    def cy(self) -> float:
        return self.height / 2

    def rays(self) -> np.ndarray:
        """Unit ray directions in world frame, shape (H*W, 3), row-major."""
        u = np.arange(self.width) + 0.5
        v = np.arange(self.height) + 0.5
        uu, vv = np.meshgrid(u, v)
        d = np.stack([(uu - self.cx) / self.f, np.ones_like(uu), -(vv - self.cy) / self.f], axis=-1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return d.reshape(-1, 3)

    def unproject(self, u: np.ndarray, v: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Pixel centres (u, v) with depth to world points, shape (..., 3)."""
        x = (np.asarray(u, dtype=np.float64) + 0.5 - self.cx) / self.f * depth
        z = -(np.asarray(v, dtype=np.float64) + 0.5 - self.cy) / self.f * depth
        p = self.position
        return np.stack([x + p[0], depth + p[1], z + p[2]], axis=-1)

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World points to (u, v) pixel coordinates (continuous) and depth."""
        p = np.asarray(pts, dtype=np.float64) - np.asarray(self.position)
        depth = p[..., 1]
        u = p[..., 0] / depth * self.f + self.cx - 0.5
        v = -p[..., 2] / depth * self.f + self.cy - 0.5
        return u, v, depth

    def points(self, depth: np.ndarray) -> np.ndarray:
        """A depth image to world points, shape (H, W, 3)."""
        vv, uu = np.mgrid[0 : self.height, 0 : self.width]
        return self.unproject(uu, vv, depth)


def scene_camera() -> Camera:
    c = config.robot()["camera"]
    return Camera(c["scene"][0], c["scene"][1], tuple(c["position"]), c["hfov_deg"])


def monitor_camera() -> Camera:
    c = config.robot()["camera"]
    return Camera(c["monitor"][0], c["monitor"][1], tuple(c["position"]), c["hfov_deg"])


def quat_from_z(direction: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) rotating local +z onto `direction`."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, d)
    q = np.array([1.0 + c, axis[0], axis[1], axis[2]])
    return q / np.linalg.norm(q)


def depth_normals(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Surface normals from a grid of points by central differences, oriented toward the camera (-y)."""
    dx = np.zeros_like(points)
    dz = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]  # toward +u
    dz[1:-1, :] = points[:-2, :] - points[2:, :]  # toward -v, i.e. up
    n = np.cross(dz, dx)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(norm > 1e-9, n / np.maximum(norm, 1e-9), 0.0)
    flip = n[..., 1] > 0
    n[flip] *= -1
    ok = valid.copy()
    ok[:, 0] = ok[:, -1] = False
    ok[0, :] = ok[-1, :] = False
    ok[:, 1:-1] &= valid[:, 2:] & valid[:, :-2]
    ok[1:-1, :] &= valid[2:, :] & valid[:-2, :]
    n[~ok] = 0.0
    return n
