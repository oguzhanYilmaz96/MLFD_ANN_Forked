from .aux_methods import area_rectangle


def test_generate_points(area_rectangle):
    """Generate a 100 points. Check if the output variables have the appropriate lengths."""
    area_rectangle.generate_points(n=100)

    assert len(area_rectangle.x) == 100
    assert len(area_rectangle.y) == 100
