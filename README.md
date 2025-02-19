# PyLandscape

## Introduction

`pylandscape` is a pytorch library for Hessian based analysis of neural network models. The library enables computing the following metrics:

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

### Install from source

You can also compile the library from source

```
git clone https://github.com/balditommaso/PyLandscape.git
pip install -r requirements.txt
```

### Download the HGCAL dataset

You can download the dataset for the ECON-T model

```
wget -P ./data/ECON/ https://retis.santannapisa.it/~tbaldi/hgcal_dataset/hgcal22data_signal_driven_ttbar_v11.tar.gz 
tar -xvf ./data/ECON/hgcal22data_signal_driven_ttbar_v11.tar.gz -C ./data/ECON
mv ./data/ECON/hgcal22data_signal_driven_ttbar_v11/nElinks_5/*.csv ./data/ECON/
```

### Download the Fusion dataset

Soon available!

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
    --config ./config/econ/baseline.yml \
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
```

## Citation

PyLandscape has been developed as part of the following paper. We appreciate it if you would please cite the following paper if you found the library useful for your work:

* T. Baldi, J. Campos, O. Weng, C. Geniesse, N. Tran, R. Kastner, A. Biondi. Loss Landscape Analysis for Reliable Quantized ML Models for Scientific Sensing, 2025, [PDF](http://arxiv.org/abs/2502.08355).
