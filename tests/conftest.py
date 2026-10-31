import numpy as np
import pytest

from podpick.sim.scene import sample_scene
from podpick.sim.world import World


class Oracle:
    """Perception from the truth, for tests of everything downstream of perception."""

    def __init__(self, w: World):
        self.w = w

    def __call__(self, obs):
        seg = self.w.labels()["seg"]
        t = self.w.target_index
        occ = np.zeros(3)
        occ[self.w.occlusion_class(t, seg)] = 1.0
        return {"mask": seg == t, "occlusion": occ}


@pytest.fixture
def world():
    w = World(sample_scene("mixed", 1000017))
    w.set_target(5)
    return w
