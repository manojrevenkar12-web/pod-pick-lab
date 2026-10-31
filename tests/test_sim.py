import numpy as np

from podpick.sim.scene import sample_scene
from podpick.sim.world import World


def test_scenes_are_reproducible():
    a, b = sample_scene("mixed", 42), sample_scene("mixed", 42)
    assert a.to_dict() == b.to_dict()
    assert sample_scene("mixed", 43).to_dict() != a.to_dict()


def test_items_settle_inside_the_bin():
    w = World(sample_scene("dense", 7))
    poses = w.item_poses()
    assert w.present.sum() >= len(poses) - 1
    inside = poses[w.present]
    assert (inside[:, 1] > -0.01).all() and (inside[:, 2] > -0.01).all()


def test_observation_carries_no_truth(world):
    o = world.observe()
    assert o.depth.shape == (96, 128) and o.intensity.shape == (96, 128)
    assert set(vars(o)) == {"depth", "intensity", "query", "band_z"}
    assert set(vars(o.query)) == {"dims", "albedo", "mass"}


def test_snapshot_restore_replays_identically(world):
    snap = world.snapshot()
    d1 = world.observe().depth
    world.open_band()
    world.place_tool(np.array([0.0, -0.1, 0.1]), np.array([0, 1.0, 0]))
    world.move(np.array([0.0, 0.05, 0.1]), 0.3)
    p1 = world.item_poses()
    world.restore(snap)
    d2 = world.observe().depth
    world.open_band()
    world.place_tool(np.array([0.0, -0.1, 0.1]), np.array([0, 1.0, 0]))
    world.move(np.array([0.0, 0.05, 0.1]), 0.3)
    assert np.array_equal(d1, d2)
    assert np.allclose(world.item_poses(), p1)


def test_suction_holds_a_light_box_by_its_front_face(world):
    t = world.target_index
    b = world.item_body[t]
    it = world.spec.items[t]
    p = world.data.xpos[b].copy()
    R = world.data.xmat[b].reshape(3, 3)
    front = p - np.array([0, abs(R[1, :]) @ np.array(it.half), 0])
    world.open_band()
    world.place_tool(front + [0, -0.05, 0], np.array([0, 1.0, 0]))
    for _ in range(10):
        tk = world.move(front + [0, 0.01, 0], 0.2, stop_on_contact=True)
        if tk.contact:
            break
    assert tk.contact
    assert world.vacuum(True) > 0.5
    for _ in range(40):
        if world.move(np.array([front[0], -0.25, front[2]]), 0.4).reached:
            break
    world.settle(0.3)
    o = world.outcome()
    assert o["target_out"] and o["held"] == t


def test_no_seal_on_air(world):
    world.place_tool(np.array([0.0, -0.2, 0.1]), np.array([0, 1.0, 0]))
    assert world.vacuum(True) < 0.1
    assert world.held_item() is None
