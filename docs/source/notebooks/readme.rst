UTPNerves - Preprocessing
=========================

Installation
------------

.. code:: ipython3

    !pip install -U git+https://github.com/ProyectoNervios/python-preprocessing.git

Usage
-----

.. code:: ipython3

    from utpnerves.preprocessing import Process
    
    p = Process()
    clean image = p.transform(raw_array)
