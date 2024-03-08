from multiprocessing.spawn import freeze_support
import matplotlib.pyplot as plt
from montecarlo import MonteCarloIntegrator

def g(x,y):
    return x**2*y**3-(x**2+y**2-1)**3

def f(x,y):
    return 1


if __name__ == '__main__':
    freeze_support()
    # integrator = MonteCarloIntegrator(rectangle=(1.9,4.1,3.9,6.1))
    integrator = MonteCarloIntegrator(rectangle=(-1.5, 1.5, -1.5, 1.5), parallel=True)
    integrator.g = g
    integrator.f = f
    integrator.generate_points(n=2000)
    integrator.plot()
    plt.show()
    integrator.compute_integral()
    print(integrator.integral)
