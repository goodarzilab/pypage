from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
   name='bio-pypage',
   version='0.1.4',
   description='Python implementation of the PAGE algorithm',
   author='Artemy Bakulin, Noam Teyssier',
   long_description=long_description,
   long_description_content_type="text/markdown",
   url='https://github.com/goodarzilab/pypage',
   author_email="logic2000.bakulin@gmail.com",
   packages=['pypage', 'pypage.io'],
   install_requires=[
       'numpy',
       'pandas',
       'numba',
       'tqdm',
       'scipy',
       "pytest",
       "pybiomart",
       "matplotlib"],
   classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],)
