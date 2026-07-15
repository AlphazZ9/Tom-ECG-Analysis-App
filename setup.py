"""
setup.py -- minimal setup for development install.

    pip install -e .

This makes `python -m ecg` and `import ecg` work from anywhere.
"""
from setuptools import setup, find_packages

setup(
    name="ecg-analysis",
    version="6.0.0",
    description="Mouse cardiac ECG analysis toolkit",
    packages=find_packages(include=["ecg", "ecg.*"]),
    python_requires=">=3.10",
    install_requires=[
        "customtkinter",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "neurokit2",
        "openpyxl",
        "Pillow",
        "scikit-learn",
        "joblib",
    ],
    extras_require={
        "h5": ["h5py"],
        "prism": ["pzfx"],
    },
    entry_points={
        "console_scripts": [
            "ecg-analysis=ecg.__main__:main",
        ],
    },
)
