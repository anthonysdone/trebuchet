import sys
from setuptools import setup, find_packages # type: ignore

setup(
    name="trebuchet",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
)
