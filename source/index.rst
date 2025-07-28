.. general-machine-learning documentation master file, created by
   sphinx-quickstart on Sun Jul 27 20:06:08 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

general-machine-learning documentation
======================================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.

to build, run ``sphinx-build source build``
requires standalone pandoc to be installed (not pip-installed)
still need to figure out how to get make to run on Windows
it should have ``nmake`` available out of the box
and I downloaded ``make`` from GNU
Could just create a ``launch.json`` to do it

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   placeholder/markdown.md
   placeholder/notebook.ipynb
   placeholder/restructuredtext.rst

.. toctree::
   :caption: DataCamp:
   :glob:

   datacamp/**
