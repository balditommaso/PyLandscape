from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()
    
    
setup(
    name="pylandscape",
    version="0.0.15",
    description="Python package to explore the loss landscape of Machine Learning models",
    packages=["pylandscape", "pylandscape.mc_utils"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/balditommaso/PyLandscape",
    download_url="https://github.com/balditommaso/PyLandscape/archive/refs/tags/main.tar.gz",
    author="Tommaso Baldi",
    author_email="tommaso.baldi@santannapisa.it",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 3 - Alpha"
    ],
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
        "PyHessian"
    ],
    extras_require={
        "dev": ["twine>=4.0.2"]
    },
    python_requires=">=3.8"
)