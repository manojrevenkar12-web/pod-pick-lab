import numpy as np

from podpick.grasp.candidates import FEATURES, generate, heuristic_scores


def test_candidates_lie_on_the_target_and_face_the_camera(world):
    obs = world.observe()
    seg = world.labels()["seg"]
    mask = seg == world.target_index
    c = generate(obs, mask, np.array([1.0, 0, 0]))
    assert 0 < len(c) <= 16
    assert mask[c.pixels[:, 0], c.pixels[:, 1]].all()
    assert (c.normals[:, 1] < -0.5).all()
    assert c.crops.shape == (len(c), 3, 32, 32)
    assert c.features.shape == (len(c), len(FEATURES)) and np.isfinite(c.features).all()
    # the crops are centred on the candidate: the centre pixel is on the target mask
    assert (c.crops[:, 1, 16, 16] == 1).all()


def test_heuristic_prefers_free_space(world):
    obs = world.observe()
    mask = world.labels()["seg"] == world.target_index
    c = generate(obs, mask, np.array([1.0, 0, 0]))
    s = heuristic_scores(c)
    assert c.clearance[np.argmax(s)] >= np.median(c.clearance) - 1e-9


def test_no_mask_no_candidates(world):
    obs = world.observe()
    c = generate(obs, np.zeros((96, 128), bool), np.array([1.0, 0, 0]))
    assert len(c) == 0
