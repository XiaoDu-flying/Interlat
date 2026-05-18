#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hidden-state-training",
    version="1.0.0",
    author="Hidden State Training Team",
    author_email="contact@hidden-state-training.com",
    description="A comprehensive framework for training language models enhanced with hidden state representations",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/XiaoDu-flying/Interlat",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
            "flake8>=5.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
            "sphinx-autodoc-typehints>=1.19.0",
        ],
        "full": [
            "flash-attn>=2.0.0; sys_platform == 'linux'",
            "triton>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hst-collect=data_collection.main:main",
            "hst-train=core_training.train:main",
            "hst-setup=scripts.setup:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt"],
    },
    keywords=[
        "machine learning",
        "deep learning",
        "natural language processing",
        "hidden states",
        "language models",
        "transformers",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/XiaoDu-flying/Interlat/issues",
        "Source": "https://github.com/XiaoDu-flying/Interlat",
    },
)