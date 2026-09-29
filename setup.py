"""Minimal setup.py for legacy installers."""
from setuptools import setup, find_packages

setup(
    name="terraforge",
    version="1.0.0",
    description="Spatial multimodal world model framework for the physical world",
    packages=find_packages(include=["terraforge*"]),
    python_requires=">=3.8",
    install_requires=[],
)
