# fashion-tagger
[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://GitHub.com/cenkcorapci/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/cenkcorapci/fashion-tagger.svg)

Multi-label classification based on [Fashion Product Images Data Set](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)

- The data set contains attributes in 6 different categories like usage and season.
- There are not enough samples to learn every single one of them so i omitted any label that has below 2000 samples. This model tries to predict remaining 24 labels.
- Used [iamgaug](https://github.com/aleju/imgaug) for image augmentation.
- Model is a [DenseNet201](https://arxiv.org/abs/1608.06993) pre trained on [Image Net](http://www.image-net.org/) with and added dense selu layer and a dropout layer.
- Used hamming loss and sigmoid activations for each label.

| Validation Metric               | Score  |
|---------------------------------|--------|
| Categorical accuracy            | 0.6138 |
| Accuracy                        | 0.6053 |

## Usage

Download the [data set](https://www.kaggle.com/paramaggarwal/fashion-product-images-small) and set
the paths in *commons/config.py*. Then run experiment.py for training,

```bash
python experiment.py
```