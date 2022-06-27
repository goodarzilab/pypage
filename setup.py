from setuptools import setup

setup(
   name='pypage',
   version='0.0.9',
   description='Python implementation of the PAGE algorithm',
   author='Noam Teyssier',
   packages=['pypage'],
   install_requires=[
       'numpy',
       'pandas',
       'numba',
       'tqdm',
       'scipy',
       'sphinx',
       "numpydoc",
       "pydata-sphinx-theme",
       "sphinx-autodoc-typehints",
       "pybiomart",
   ])
