from ..montecarlo import MonteCarloIntegrator
from pytest import fixture, approx, mark


@fixture
def area_rectangle():
    """Returns a MonteCarloIntegrator object with a defined seed. f(x)=1 and g(x) is a rectangle defined [0,1]x[2x4]."""

    external_rectangle = (-1, 2, 1, 5)

    integrator = MonteCarloIntegrator(rectangle=external_rectangle, parallel=False, seed=8)

    return integrator


def g(x, y):
    # g function of a rectangle
    return (1 if (0 <= x <= 1 and 2 <= y <= 4) else -1)


def f(x, y):
    # f function of a rectangle
    return 1
