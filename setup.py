"""Setup configuration for dynClimToolkit"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dynclim",
    version="0.2.0",
    author="Michael Rosen",
    author_email="mrosena@gmail.com",
    description="Dynamic Climate Safety Analysis Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dr-rosen-rosen/DynamicNarrativeClimateToolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "matplotlib>=3.3",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "dynclim-run=scripts.run_experiment:main",
        ],
    },
)
