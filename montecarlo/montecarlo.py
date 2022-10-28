from __future__ import annotations

import warnings
from functools import partial
from itertools import product, tee
from typing import Callable

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

params = {"figure.figsize": [9, 9], "text.usetex": False, "font.size": 24}
plt.style.use(params)


class MonteCarloIntegrator:
    """
    Class to compute a 2D integral using a MonteCarlo method
    """

    def __init__(
        self, rectangle: list, parallel: bool = False, seed: bool | int = False
    ):
        """

        Parameters
        ----------
        rectangle: tuple
            with 4 elements (x0, x1, y0, y1) defining the integration rectangle R
        parallel: bool
            If true, it will use multiprocessing to evaluate each point. Otherwise, single process.
        seed: float or False
            Seed for the random number generator (repeatability). If False, random.
        """
        self.__func_g = None
        self.__func_f = None
        self.parallel = parallel
        self.x = None
        self.y = None
        self.rectangle = rectangle
        self.x0 = rectangle[0]
        self.x1 = rectangle[1]
        self.y0 = rectangle[2]
        self.y1 = rectangle[3]
        self.n = None
        self.integral = None
        if seed:
            np.random.seed(seed)

    @property
    def g(self):
        """
        Function that defines the integration region. It should be defined implicitly as G(x,y) = 0.

        Returns
        -------
        User defined function g
        """
        return self.__func_g

    @g.setter
    def g(self, func_g: Callable) -> None:
        """
        Setter for the g function. If the argument provided is not callable it will show a warning.

        Parameters
        ----------
        func_g: function
            provided by the user
        """
        if callable(func_g):
            self.__func_g = func_g
        else:
            warnings.warn("The function g is not callable, please provide a new one.")

    @property
    def f(self):
        """
        Function of x,y that is to be integrated. Defined as F(x,y) = 0.
        If 1 is provided, the integration provides the area.

        Parameters
        ----------
        User defined function f
        """

        return self.__func_f


    @f.setter
    def f(self, func_f: Callable) -> None:
        """
        Setter for the f function. If the argument provided is not callable it will show a warning.

        Parameters
        ----------
        func_f: function
            provided by the user
        """

        if callable(func_f):
            self.__func_f = func_f
        else:
            warnings.warn('The function f is not callable, please provide a new one.')

    def generate_points(self, n: int = 20):
        """
        Function to generate the random points.

        Parameters
        ----------
        n: int
        """

        self.n = n
        self.x = np.random.uniform(self.x0, self.x1, n)
        self.y = np.random.uniform(self.y0, self.y1, n)

    def compute_integral(self):
        """
        Compute integral using the MonteCarlo Method.
        Each point is computed by the static method :func:`_compute_sample`.
        Then, it applies the theory of the mean-value theorem to compute the area.
        The return value is stored within the class, and also returned.

        Returns
        --------
        Value of the integral.
        """
        iterator_data = zip(self.x, self.y)
        if self.parallel:
            pool = multiprocessing.Pool(processes=8)
            f_values = pool.map(
                partial(self._compute_sample, f=self.f, g=self.g), iterator_data
            )
        else:
            f_values = list(
                map(partial(self._compute_sample, f=self.f, g=self.g), iterator_data)
            )

        iterator_data = zip(self.x, self.y)
        if self.parallel:
            pool = multiprocessing.Pool(processes=8)
            f_values = pool.map(partial(self._compute_sample, f=self.f, g=self.g), iterator_data)
        else:
            f_values = list(map(partial(self._compute_sample, f=self.f, g=self.g), iterator_data))
        f_values = np.array(f_values)
        f_values = f_values[f_values != np.array(None)]
        num_inside = len(f_values)
        f_mean = np.mean(f_values)
        area = num_inside / float(self.n) * (self.x1 - self.x0) * (self.y1 - self.y0)
        self.integral = area * f_mean
        return self.integral

    @staticmethod
    def _compute_sample(point: tuple, f: Callable, g: Callable) -> float | None:
        """
        Computes the value of the function f in the point x, y.
        If the point is inside of g, the function returns the value of the function in f.
        Otherwise, the function returns None.

        Parameters
        ----------
        point: tuple
            x, y pair of values
        f: function
            callable function defining the function to be integrated
        g: function
            callable function defining the region

        Returns
        -------

        """
        x = point[0]
        y = point[1]
        if g(x, y) >= 0:
            return f(x, y)
        else:
            return None

    @staticmethod
    def _get_nproc() -> int:
        """
        Get the number of processors available for parallel computation using multiprocessing

        Returns
        -------
        number of processors
        """

        #### TODO: YOUR CODE GOES HERE #####
        raise NotImplementedError

    def plot(self, all_random_points: bool = False):
        """
        Function to plot the process.
        In general, it will plot the contour plot filled of f, the domain defined by g with a shadowed area and border, and the montecarlo points.
        If any of these is not available, it will simply skip it.
        Uses two auxiliary functions to actually plot.


        Parameters
        ----------
        all_random_points: bool
            If True, it will plot all the points. This may be slow if n is large.

        Returns
        -------
        fig: figure handles
        ax: axes handles

        """
        # Draw n random points in a rectangle
        # this is only for visualization, to make the graph for the course

        fig, ax = plt.subplots()
        self._plot_f_g(ax=ax, filled=True, function=self.f, alpha=0.4, cmap="viridis")
        self._plot_f_g(ax=ax, filled=False, function=self.g, alpha=1, colors="k")
        self._plot_f_g(
            ax=ax, filled=True, levels=[0, 1], function=self.g, alpha=0.6, cmap="jet"
        )
        self._plot_random_points(ax=ax, all_random_points=all_random_points)
        return fig, ax

    def _plot_f_g(self, function: Callable, ax, filled: bool, **kwargs):
        """Auxiliary function to plot f or g. It uses the vectorized function of numpy to be able to plot.

        Parameters
        ----------
        function: callable function.
            To be plotted
        ax: axis handles
            Where to plot the data.
        filled: bool
            If filled, it will use a contourf (typically for f function), otherwise it will use a normal contour, only keeping level 0 to draw the region.
        kwargs
            Any kwargs that can be passed to pyplot
        """
        if function is None:
            print("A function has not been provided...")
            return

        if function is None:
            print('A function has not been provided...')
            return
        x1 = np.linspace(self.x0, self.x1, 500)
        y1 = np.linspace(self.y0, self.y1, 500)
        X, Y = np.meshgrid(x1, y1)
        Z = np.vectorize(function)(X, Y) # vectorize is used to properly apply the function f or g
        if filled:
            ax.contourf(X, Y, Z, **kwargs)
        else:
            ax.contour(X, Y, Z, [0],  **kwargs)

    def _plot_random_points(self, ax, all_random_points, **kwargs):
        """
        Auxiliary function to plot the random montecarlo points

        Parameters
        ----------
        ax: axis handles
            Where to plot the data.
        all_random_points: bool
            If True, it will plot all the points. This may be slow if n is large.\
        kwargs
            Any kwargs that can be passed to pyplot

        """

        # check if the numbers have been generated
        ### TODO: YOUR CODE HERE

        if (len(self.x) > 3000) and (not all_random_points):
            print(
                "This is a lot of points...I will reduce it to 3000 for the visualization"
            )
            npoints = 3000  # take the first 3000 points
        else:
            npoints = None  # take all the points.

        iterator_data = zip(self.x[:npoints], self.y[:npoints])
        x, y = zip(*iterator_data)
        ax.scatter(x, y, alpha=0.7, **kwargs)
