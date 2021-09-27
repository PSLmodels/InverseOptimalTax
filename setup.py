from setuptools import setup

with open("README.md") as f:
    longdesc = f.read()

version = "0.0.0"

config = {
    "description": "Inverse Optimal Tax Model",
    "url": "https://github.com/PSLmodels/InverseOptimalTax",
    "download_url": "https://github.com/PSLmodels/InverseOptimalTax",
    "description": "iot",
    "long_description": longdesc,
    "version": version,
    "license": "CC0 1.0 Universal public domain dedication",
    "packages": ["iot"],
    "include_package_data": True,
    "name": "iot",
    "install_requires": ["numpy", "pandas", "scipy"],
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: CC0 1.0 Universal public domain dedication",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    "tests_require": ["pytest"],
}

setup(**config)
