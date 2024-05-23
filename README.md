# EE340-Project2-DAGMM

## Introduction

Repo for EE340 Statistical Learning for Data Science Project 2, SUSTech, 2023 Spring.

For the project requirement, please refer to the [requirement](./requirement.pdf). I am sorry that the requirement is only in Chinese.

We choose item 2 (Anomaly Detection 异常检测) and implement the DAGMM model with PyTorch.

DAGMM (Deep Autoencoding Gaussian Mixture Model) is an unsupervised anomaly detection model proposed by Zong et al. in 2018. 

> "Instead of using decoupled two-stage training and the standard Expectation-Maximization (EM) algorithm, DAGMM jointly optimizes the parameters of the deep autoencoder and the mixture model simultaneously in an end-to-end fashion, leveraging a separate estimation network to facilitate the parameter learning of the mixture model."

## Usage
```bash
git clone https://github.com/squarezhong/EE340-Project2-DAGMM
cd EE340-Project2-DAGMM
python main.py
```

It is optional to use `pip install -r requirements.txt` to install the required packages.

#### Hyperparameters
The hyperparameters are not fine-tuned (even not tuned at all orz). You can change the hyperparameters in `main.py` to get better performance.

## Reference
[Zong, B., Song, Q., Min, M. R., Cheng, W., Lumezanu, C., Cho, D., & Chen, H. (2018). Deep autoencoding Gaussian mixture model for unsupervised anomaly detection. International Conference on Learning Representations (ICLR).](https://bzong.github.io/doc/iclr18-dagmm.pdf)