from .aux_methods import area_rectangle, f, g
from pytest import approx

def test_integral_rectangle(area_rectangle):
    """Test for the area of the rectangle. Use tolerance with abs +- 0.1."""
    area_rectangle.generate_points(n=3000)
    area_rectangle.f = f
    area_rectangle.g = g
    value_area = area_rectangle.compute_integral()
    assert approx(value_area, abs=0.2) == 2
