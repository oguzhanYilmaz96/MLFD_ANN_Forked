=============================
Troubleshooting for Windows
=============================

In Windows, the :py:class:`~multiprocesssing` library may give certain issues.
In order to avoid this, we need to add the following. In the main file we add this import:

.. code-block:: python

    from multiprocessing.spawn import freeze_support

Then, we define the main part of the script and add the following lines:

.. code-block:: python

    if __name__ == '__main__':
        freeze_support()

The rest of the code can continue as normal below as in :ref:`Usage`.