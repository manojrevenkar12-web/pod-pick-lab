import numpy as np

from podpick.geometry import Camera, depth_normals, quat_from_z, scene_camera


def test_project_unproject_roundtrip():
    cam = scene_camera()
    u, v = np.array([3, 64, 120]), np.array([5, 48, 90])
    d = np.array([0.4, 0.55, 0.7])
    pts = cam.unproject(u, v, d)
    uu, vv, dd = cam.project(pts)
    assert np.allclose(uu, u) and np.allclose(vv, v) and np.allclose(dd, d)


def test_quat_from_z_maps_z_onto_direction():
    import mujoco

    for d in ([0, 1, 0], [0.3, 0.9, -0.2], [0, 0, -1]):
        q = quat_from_z(np.array(d, float))
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, q)
        assert np.allclose(m.reshape(3, 3)[:, 2], np.array(d) / np.linalg.norm(d), atol=1e-9)


def test_normals_of_a_wall_face_the_camera():
    cam = Camera(32, 24, (0.0, -0.4, 0.1), 46.0)
    depth = np.full((24, 32), 0.6)
    n = depth_normals(cam.points(depth), depth > 0)
    inner = n[2:-2, 2:-2].reshape(-1, 3)
    assert np.allclose(inner, [0, -1, 0], atol=1e-6)
