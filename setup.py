from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()
    
    
setup(
    name="pylandscape",
    version="0.0.10",
    description="Pythone package to explore the loss landscape of Machine Learning models",
    package_dir={"": "pylandscape"},
    packages=find_packages(where="pyhessian"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/balditommaso/PyLandscape",
    author="Tommaso Baldi",
    author_email="tommaso.baldi@santannapisa.it",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_require={
        "dev": ["twine>=4.0.2"]
    },
    python_requires=">=3.8"
)