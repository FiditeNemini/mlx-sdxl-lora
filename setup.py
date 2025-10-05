#!/usr/bin/env python3
"""
Setup script for the VLM-Powered Image Captioning Tool
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlx-sdxl-lora-caption-tool",
    version="1.0.0",
    author="Fidite Nemini",
    description="VLM-Powered Image Captioning Tool for LoRA Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FiditeNemini/mlx-sdxl-lora",
    py_modules=["caption_tool", "example_usage"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "gradio>=4.0.0",
        "mlx-vlm>=0.0.1",
        "mlx>=0.0.1",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "caption-tool=caption_tool:main",
        ],
    },
)
