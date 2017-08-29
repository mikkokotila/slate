---
title: Autonomio Deep Learning Workbench User Manual

language_tabs: # must be one of https://git.io/vQNgJ
  - python

toc_footers:
  - <a href='https://github.com/autonomio/core-module'>Autonomio on Github</a> 
  - <a href='https://github.com/autonomio/core-module#fork-destination-box'>For Autonomio</a>
  - <a href='http://autonom.io'>Autonomio Website</a> 

includes:
  - commands
  - examples
  - shapes
  - vectorize

search: true
---

# Introduction

Autonomio provides a high-level abstraction layer to building, configuring and optimizing neural networks and then using the trained models to make predictions in any environment. Unlike with other similar solutions, there is no need for signing up, API keys, cloud instances, or GPUs, and you have 100% control over the model. A typical installation takes a minute, and training a model not more than few minutes including data transformation from raw dataset with even thousands of columns, open text, and unstructured labels. Nothing is pre-trained, and only you have access to your data and predictions. There is no commercial entity behind Autonomio, but a non-profit research Foundation.

This document covers the functionality of Autonomio. If you're looking for a high level overview of the capabilities, you might find the [Autonomio website](http://autonom.io) more useful. 

# 1-Minute Pipepline

> To train a model use this code

```python
# do the python imports 
from autonomio.commands import data, wrangler, train, predictor
%matplotlib inline

# import the data from csv
df = data('medicare_10k.csv', mode='file', header=None)

# preprocess the data
df = wrangler(df,'z')

# train a neural net
train([2,17],'z',df,epoch=20,loss='logcosh',flatten='median')
```

> NOTE: list of column index can be used with 3 or more columns. Using two integers will be considered a range of columns.

Autonomio is very easy to use and it's very easy to memorize the namespace which is just 4 commands and less than 40 arguments combined. Namespace memorization is one of the key differences between advanced and beginner users. Whereas Autonomio helps lower skill level practitioners to dramatically improve their capability, advanced practitioners enjoy significant productivity gains and headache reduction.

<aside class="notice">
You must replace <code>medicare_10k.csv</code> with your own dataset.
</aside>

# Installation

The simplest way is to get the latest well tested version is to install with pip from the repo directly. This way you get the latest well tested version, with the latest features. 

<code>pip install git+https://github.com/autonomio/core-module.git</code>


# Training Neural Network

> A typical use of the training function for - you guessed it - training a neural network. 

```python

train([1,25],'Survived',df,
                        flatten='none',
                        epoch=250,
                        dropout=0,
                        batch_size=batch,
                        loss='logcosh',
                        activation='elu',
                        layers=layer,
                        shape=shape,
                        verbose=0)
```

Autonomio provides a very high level abstraction layer to several deep learning models:

- Multi Layer Perceptors (MLP)
- LSTM
- Regression

These are all accessed through the train() command. 