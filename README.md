# PyLandscape

## Introduction
PyLandscape is a pytorch library for Hessian based analysis of neural network models. The library enables computing the following metrics:

- [CKA similarity]()
- [Hessian metrics](https://arxiv.org/pdf/1912.07145)
- [Mode connectivity]()
- [Loss surface]()

*NOTE*: All the functionalities relative to the computation of the Hessian metrics have been embedded via [PyHessian](https://github.com/amirgholami/PyHessian). If your interested in learning more about how these metrics are computed have a look to their Repository.


## Usage
### Install from Pip
You can install the library from pip (soon available)
```
pip install pylandscape
```

### Install from source
You can also compile the library from source
```
git clone https://github.com/balditommaso/PyLandscape.git
python setup.py install
```

TODO: add usage

## Citation
PyLandscape has been developed as part of the following paper. We appreciate it if you would please cite the following paper if you found the library useful for your work:

* T. Baldi, J. Campos, O. Weng, C. Geniesse, N. Tran, R. Kastner, A. Biondi. Loss Landscape Analysis for Reliable Quantized ML Models for Scientific Sensing, 2025, [PDF]().
