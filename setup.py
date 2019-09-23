from setuptools import setup
import re

setup(
    name="xpcapy",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3"
    ],
    description="A package for mixed variable types",
    install_requires=[
        "numpy",
        "scipy"
    ],
    long_description="""XPCA is a tool for decomposing matrices of data with mixed variable types 
    (e.g., continuous, count, binary, etc.)
    This python version is maintained by a team at Sandia National Labs. \n""",
    maintainer="XPCA team",
    maintainer_email="xpca@sandia.gov",
    version=re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        open(
            "xpcapy/__init__.py",
            "r").read(),
        re.M).group(1),
    packages=['xpcapy']
)
