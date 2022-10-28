from .aux_methods import area_rectangle


def test_check_rectangle_dimensions(area_rectangle):
    assert area_rectangle.x0 == -1
    assert area_rectangle.x1 == 2
    assert area_rectangle.y0 == 1
    assert area_rectangle.y1 == 5