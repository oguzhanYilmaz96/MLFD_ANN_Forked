from pytest import fixture, approx, mark
from .aux_methods import area_rectangle, f, g


@mark.parametrize("point,expected", [
    ((0, 0), None),
    ((0.5, 2.5), 1),
    ((0.75, 3), 1)
])
def test_compute_sample_point(area_rectangle, point, expected):
    """Test if the code returns the correct value for parametrized test"""
    value_point = area_rectangle._compute_sample(point, f=f, g=g)
    assert value_point == expected