from setuptools import setup

setup(
   name='pypage',
   version='0.0.4',
   description='Python implementation of the PAGE algorithm',
   author='Noam Teyssier',
   packages=['pypage'],
   install_requires=['numpy', 'pandas', 'numba', 'tqdm', 'scipy']
)
