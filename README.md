# PyLandscape

## Introduction

`pylandscape` is a Pytorch library for loss landscape analysis of neural networks. The library enables computing the following metrics:

- [CKA similarity](https://arxiv.org/pdf/2010.15327)
- [Hessian metrics](https://arxiv.org/pdf/1912.07145)
- [Mode connectivity](https://arxiv.org/pdf/1802.10026)
- [Loss surface](https://arxiv.org/pdf/1712.09913)

*NOTE*: All the functionalities relative to the computation of the Hessian metrics have been embedded via [PyHessian](https://github.com/amirgholami/PyHessian). If your interested in learning more about how these metrics are computed have a look to their Repository.

## Usage

### Install from Pip

You can install the library from pip:

```
pip install pylandscape
```
<!-- 
### Install from source

You can also compile the library from source

```
git clone https://github.com/balditommaso/PyLandscape.git
pip install -r requirements.txt
```

### Download the HGCAL dataset

Hide for double blinded peer reviews.


### Download the Fusion dataset

Hide for double blinded peer reviews.

### Train the models

1. Train full precision (FP32) version of the model:

```
. scripts/train.sh \
    --config ./config/econ/baseline.yml \
    --bs 1024 \
    --lr 0.0015625 \
    --device_id 0 \
    --num_test 3 \
    --full_precision
```

2. Fine tune the models with QAT:

```
. scripts/train.sh \
    --config ./config/large_econ/baseline_gaussian.yml \
    --bs 1024 \
    --lr 0.0015625 \
    --device_id 0 \
    --num_test 3 \
    --pretrained
```

3. Test the model both metrics and benchmarks
```
. scripts/test.sh \
    --config ./config/econ/baseline.yml \
    --bs 1024 \
    --lr 0.0015625 \
    --device_id 0 \
    --max_processes 3 \
    --num_models 3
``` -->

## Research
If you used `Pylandscape` consider to cite:
```
@misc{baldi2025losslandscapeanalysisreliable,
      title={Loss Landscape Analysis for Reliable Quantized ML Models for Scientific Sensing}, 
      author={Tommaso Baldi and Javier Campos and Olivia Weng and Caleb Geniesse and Nhan Tran and Ryan Kastner and Alessandro Biondi},
      year={2025},
      eprint={2502.08355},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.08355}, 
}
```
