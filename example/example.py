from multiprocessing.spawn import freeze_support
import matplotlib.pyplot as plt
from montecarlo import MonteCarloIntegrator

def g_rectangle(x,y):
    return (1 if (0 <= x <= 2 and 3 <= y <= 4.5) else -1) # rectangle

def g_circle(x,y):
    return 1 if ((x-1.5)**2+(y-3.5)**2) < 1 else -1 # circle

def f(x,y):
    return 1


if __name__ == '__main__':
    freeze_support()
    # integrator = MonteCarloIntegrator(rectangle=(1.9,4.1,3.9,6.1))
    integrator = MonteCarloIntegrator(rectangle=(0, 3, 2, 5), parallel=True)
    integrator.g = g_rectangle
    integrator.f = f
    integrator.generate_points(n=6000)
    integrator.plot()
    plt.show()
    integrator.compute_integral()
    print(integrator.integral)
