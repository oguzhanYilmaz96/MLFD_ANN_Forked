======
Usage
======

Start by importing MonteCarlo Integral VKI integrator class.

.. code-block:: python

    from montecarlo import MonteCarloIntegrator

Then, we define the equation of the function :math:`f(x,y)` to be integrated. This function defines a surface. In this example, we keep it equal to 1 for simplicity.

.. code-block:: python

    def f(x,y):
        return 1

Following, we define the integration region using a function in implicit form. Let us integrate a circle of radius 1:

.. code-block:: python

    def g_circle(x,y):
        return 1 if ((x-1.5)**2+(y-3.5)**2) < 1 else -1 # circle definition

Next, we instantiate a :class:`MonteCarloIntegrator` object. We define the rectangle around in which the circle is inscribed.
We set parallel to False. If set to True, it will use the number of processors available in :class:`multiprocessing.cpu_count()`.
In this case, the seed is set to 8 for repeatability, but normally it defaults to False.

.. code-block:: python

    integrator = MonteCarloIntegrator(rectangle=(0, 3, 2, 5), parallel=False, seed=8)

Once the integrator object is created, we can link the functions for the surface and the circumference:

.. code-block:: python

    integrator.g = g_rectangle
    integrator.f = f

Now, we can generate the random MonteCarlo points:

.. code-block:: python

    integrator.generate_points(n=6000)

Before performing the integration, we may want to visualize the region to ensure we are plotting the correct areas:

.. code-block:: python

    fig, ax = integrator.plot()
    fig.savefig('regions.png')

Finally, we integrate:

.. code-block:: python

    integrator.compute_integral()
