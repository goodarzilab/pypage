from pathlib import Path
from setuptools import setup

ROOT = Path(__file__).resolve().parent


def _read_text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def _read_requirements():
    req_path = ROOT / "requirements.txt"
    if not req_path.exists():
        return []
    requirements = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


long_description = _read_text("README.md")

setup(
   name='bio-pypage',
   version='0.2.1',
   description='Python implementation of the PAGE algorithm',
   author='Artemy Bakulin, Noam Teyssier',
   long_description=long_description,
   long_description_content_type="text/markdown",
   url='https://github.com/goodarzilab/pyPAGE',
   author_email="logic2000.bakulin@gmail.com",
   packages=['pypage', 'pypage.io'],
   install_requires=_read_requirements(),
   entry_points={
        'console_scripts': [
            'pypage=pypage.cli:main',
            'pypage-sc=pypage.cli_sc:main',
        ],
   },
   classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],)
