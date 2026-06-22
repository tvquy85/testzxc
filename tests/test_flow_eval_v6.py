import numpy as np

from src.reward.evaluate_flow_v6 import metric_wins, project_rows_to_simplex


def test_metric_wins_requires_two_of_three():
    assert metric_wins({"rank": True, "pair": False, "top_decile": True})
    assert not metric_wins({"rank": True, "pair": False, "top_decile": False})


def test_project_rows_to_simplex_sums_to_one():
    out = project_rows_to_simplex(np.array([[0.2, 0.3, 0.4, 0.1, -0.2], [2.0, -1.0, 0.0, 0.0, 0.0]]))

    assert out.shape == (2, 5)
    assert np.all(out >= 0.0)
    assert np.allclose(out.sum(axis=1), 1.0)
