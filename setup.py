from setuptools import setup, find_packages

setup(
    name="autovar",
    version="0.2.0",
    description="A DSL for variance-aware probabilistic programming",
    author="AutoVar Authors",
    packages=find_packages(exclude=["tests", "demos", "testing"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "jax": [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
        ],
        "dev": [
            "pytest",
            "jupyter",
            "matplotlib",
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
