from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
   name='bio-pypage',
   version='1.0.0',
   description='Python implementation of the PAGE algorithm',
   author='Artemy Bakulin, Noam Teyssier',
   long_description=long_description,
   long_description_content_type="text/markdown",
   url='https://github.com/noamteyssier/pypage',
   author_email="logic2000.bakulin@gmail.com",
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
       "pytest",
       "pybiomart",
       "matplotlib"],
   classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],)
