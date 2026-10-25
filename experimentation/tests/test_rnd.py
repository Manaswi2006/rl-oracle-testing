import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.vlm.rnd import RND


def test_novelty_decreases_for_visited_states_and_stays_high_for_new_ones():
    rnd = RND(6, seed=0, update_every=2, batch=64)
    rs = np.random.RandomState(0)
    familiar = rs.randn(50, 6) * 0.3
    first = np.mean([rnd.step(familiar[i % 50]) for i in range(100)])
    for i in range(3000): rnd.step(familiar[i % 50])
    late = np.mean([rnd.novelty(f) for f in familiar]); novel = np.mean([rnd.novelty(rs.randn(6) * 3 + 4) for _ in range(50)])
    assert late < 0.2 * rnd.novelty(familiar[0]) + late and novel > 3 * late, (late, novel)
