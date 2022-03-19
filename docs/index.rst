.. neuroHMM documentation master file, created by
   sphinx-quickstart on Wed Jan 26 15:45:23 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to neuroHMM's documentation!
====================================

The neuroHMM project started as a student project in collaboration with the Neuroscience Institute of La Timone. The aim of the project is to provide Machine Learning tools, initially adapted to neurophysiological signals, which allow spatial, temporal and frequency analysis of experimental data. Most of the tools developed are based on hidden Markov models (HMMs), and are inspired by the `HMM-MAR library <https://github.com/OHBA-analysis/HMM-MAR>`_ developed in Matlab by a team of Oxford researchers. Our library aims to be an extension of the `scikit-learn library <https://scikit-learn.org/stable/>`_, a reference for Machine Learning experts, and follows as much as possible the scikit-learn Estimators development standards. Some of the tools developed are also based on the `hmmlearn library <https://hmmlearn.readthedocs.io/en/latest/>`_, which implements the more classical HMMs with scikit-learn development standards.

------

On this website, you will find:

* A presentation of the theory behind each of the implemented tools by following the Theory_ tab;
* A guide to installing and using the library on the `User Guide`_ page;
* Documentation of each class and associated methods in the `API Reference`_;
* Example notebooks of how to use the Estimators on the Examples_ page.

.. _Theory: theory.rst
.. _User Guide: user_guide.rst
.. _API Reference: api_reference.rst
.. _Examples: example.rst


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   theory.rst
   user_guide.rst
   api_reference.rst
   example.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
