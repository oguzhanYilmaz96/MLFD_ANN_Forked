===============
Theory
===============

In many engineering problems one needs to integrate a function in a domain. MonteCarlo integration may be an intuitive and easy solution to do so.

The idea of Monte Carlo integration of :math:`\int_a^b f(x)dx` is to use the mean-value theorem from calculus, which states that the integral
:math:`\int_a^b f(x)dx` equals the length of the integration domain, here   :math:`b-a` , times the average value of :math:`f`,   :math:`\bar{f}` ,
in :math:`[a,b]`.
The average value can be computed by sampling f at a set of random points inside the domain and take the mean of the function values.
In higher dimensions, an integral is estimated as the area/volume of the domain times the average value,
and again one can evaluate the integrand at a set of random points in the domain and compute the mean value of those evaluations.

Let us introduce some quantities to help us make the specification of the integration algorithm more precise. Suppose we have some two-dimensional integral
:math:`\int_\Omega f(x,y)dxdy`,
where :math:`\Omega` is a two-dimensional domain defined via a help function :math:`g(x,y)`:
:math:`\Omega = {(x,y)|g(x,y) > 0}`.

That is, points   :math:`(x,y)`  for which  :math:`g(x,y)>0` lie inside :math:`\Omega`, and points for which :math:`g(x,y)<\Omega`  are outside :math:`\Omega`.
The boundary of the domain   ∂Ω  is given by the implicit curve   :math:`g(x,y)=0`.
Such formulations of geometries have been very common during the last couple of decades, and one refers to g as a level-set function
and the boundary :math:`g=0` as the zero-level contour of the level-set function.
For simple geometries one can easily construct :math:`g(x,y)` by hand, while in more complicated industrial applications one must resort
to mathematical models.

Let  :math:`A(\Omega)` be the area of a domain :math:`\Omega`. We can estimate the integral by this Monte Carlo integration method:

1. embed the geometry :math:`\Omega` in a rectangular area  :math:`R`
2. draw a large number of random points    :math:`(x,y)`  in  :math:`R`
3. count the fraction  :math:`q` of points that are inside :math:`\Omega`
4. approximate  :math:`A(\Omega)/A(R)` by q, i.e., set  :math:`A(\Omega)=qA(R)`
5. evaluate the mean of :math:`f`,   :math:`\bar{f}` , at the points inside :math:`\Omega`
6. estimate the integral as   :math:`A(\Omega)\bar{f}`


Note that :math:`A(R)` is trivial to compute since :math:`R` is a rectangle,
while   :math:`A(\Omega)`  is unknown. However, if we assume that the fraction of   :math:`A(R)`  occupied by   :math:`A(\Omega)` is the same as the fraction
of random points inside :math:`\Omega`, we get a simple estimate for   :math:`A(\Omega)`.