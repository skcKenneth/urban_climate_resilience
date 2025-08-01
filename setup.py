"""
Setup script for Urban Climate-Social Network Resilience System
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="urban-climate-resilience",
    version="1.0.0",
    author="Kenneth, Sok Kin Cheng",
    author_email="sokkincheng@gmail.com",
    description="Mathematical modeling framework for urban climate-social network resilience",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skckenneth/urban_climate_resilience",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    entry_points={
        "console_scripts": [
            "climate-analysis=main:main",
            "auto-climate-analysis=auto_run:main",
        ],
    },
)
