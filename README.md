# SBRL-HAP

## Introduction

This repository contains the implementation code for paper:

**Stable Heterogeneous Treatment Effect Estimation across Out-of-Distribution Populations**

Yuling Zhang, Anpeng Wu, Kun Kuang, Liang Du, Zixun Sun, Zhi Wang

## Requirements

Python 3.6.8 with TensorFlow 1.15.0, NumPy 1.19.5, Scikit-learn 0.24.2 and MatplotLib 3.3.4.

## Instructions

`syn_data_generator.py` is an example of Synthetic Data Generation.

`sbrl_hap.py` contains the class for SBRL-HAP, which is implemented on the network backbone of the Counterfactual Regression [1].

`utils.py` includes the necessary utilities.

Run `train.py` scripts to train the model.

```python
python train.py
```

## Reference

[1] U. Shalit, F. D. Johansson, and D. Sontag, “Estimating individual treatment effect: generalization bounds and algorithms,” in Proceedings of the 34th International Conference on Machine Learning, PMLR, Jul. 2017, pp. 3076–3085. Accessed: Feb. 01, 2023.
