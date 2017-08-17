---
title: Autonomio Deep Learning Workbench User Manual

language_tabs: # must be one of https://git.io/vQNgJ
  - python

toc_footers:
  - <a href='https://github.com/autonomio/core-module'>Autonomio on Github</a> 
  - <a href='https://github.com/autonomio/core-module#fork-destination-box'>For Autonomio</a>
  - <a href='http://autonom.io'>Autonomio Website</a> 

includes:
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


# Training aNeural Network

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

Autonomio provides a very high level abstraction layer to Multilayer Percepton Neural Network with comprehensive customization of neural net from a single line command. 

# Commands 

## Train

- loss
- optimization
- activation 
- shape
- layers (even thousands of layers)
- dropout rate
- batch_size

### Data Ingestion

Compared to TensorFlow, Keras, scikit learn and other common libraries, Autonomio provides a highly convinient data ingestion function. 

- Automatically through train()
- Configured throgh train()
- Using the wrangler() utility

```python
# a single column where data is string
train('text' ,'neg', data) 

# a single column by index
train(5, 'neg', data) 

# a single column by label
train(['quality_score'], 'neg', data) 

# a range of column index
train([1,5], 'neg', data) 

# set of column labels
train(['quality_score', 'reach_score'], 'neg', data) 

# a list of column index
train([1,2,4,6,18], 'neg', data) 
```

Data can be inputted from a dataframe, or csv, txt, json or msgpack files. All common transformations take place automatically within the train() command.

- automatic transformation of dependent variables
    - from text to word vectors
        - from text labels to integers 
- automatic transformation of outcome variable
    - from continuous to categorical
        - based on mean
        - based on median 
        - based on quantiles 
        - based on ge 
    - from multi-category to binary
        - string values
        - numeric values 

Generally speaking, multilayer percepton neural nets are strongest in solving classification problems, where the outcome variable is either binary categorical (0 or 1) or multi categorical. This is why there is strong emphasis in Autonomio on making such transformations available within the train() command. 

#### BINARY (default)

- X can be text, int, or floating point 
- Y can be an int, or floating point

The default settings are optimized for making a 1 or 0 prediction and for example in the case of predicting sentiment from tweets, Autonomio gives 85% accuracy without any parameter setting for classifying tweets that rank in the most negative 20% according to NLTK Vader sentiment analysis. 

#### CATEGORICAL

- X can be text, integer 
- Y can be an integer or text
- output layer neurons must match number of categories
- change activation_out to something that works with categoricals

It's not a good idea to have too many categories, maybe 10 is pushing it in most cases. 

### Train Query Parameters

ARGUMENT | REQUIRED INPUT | DEFAULT
---------|----------------|--------
X | string, int, float | NA
Y | int,float,categorical | NA   
data | data object | NA
epoch | int | 5 
flatten | string, float | 'mean' 
dropout | float | .2
layers | int (2 through 5 | 3 
loss | string [Keras_Losses]_  | 'binary_crossentropy'
save_model | string | False 
neuron_first | int,float,categorical | 300 
neuron_last | data object | 1
batch_size | int | 10 
verbose | 0,1,2 | 0 
shape | string | 'funnel'
double_check | True or False | False 
validation | True,False,float(0 to 1)| False 

**X** = The input can be indicated in several ways::

'label'   = single column label
['a','b'] = multiple column labels
[1,12]    = a range of columns
[1,2,12]  = columns by index
The data can be multiple dtypes:
'int'     = any integer values
'float'   = any float value
'string'  = raw text or category labels

In case you need to cleanup your data first, you can do it with::

from autonomio.commands import wrangler

wrangler(data,outcome_var)

**Y** =  This can be in multiple dtype::

'int'     = any integer values
'float'   = any float value
'string'  = category labels
    
See more related to prediction variable below in the 'flatten' section.

**data** =  A pandas dataframe where you have at least one column for 'x' depedent variable (predictor) and one column for 'y' indepedent variable (prediction).

**dims** =  This is selected automatically and is not needed to worry about. NOTE: this needs to be same as x features

**epoch** = how many epocs will be run for training. More epochs will take more time.

**flatten** = For transforming y (outcome) variable. For example if the y input is continuous but prediction is binary, then a flattening of some sort should be used.

OPTIONS:  'mean','median','mode', int, float, 'cat_string', 'cat_numeric', and 'none'
        
**dropout** = The fraction of learning that will be "forgotten" on each each learning event.

**layers** = The number of dense layers the model will have. Note that each dense layer is followed by a dropout layer.

**model** = This is currently not in use. Later we add LSTM and some other model options, then it will be activated.

**loss** = The loss to be used with the model. All the Keras losses all available https://keras.io/losses/

**optimizer** = The optimizer to use with the model. All the Keras optimizers are all available > https://keras.io/optimizers/

**activation** = Activation for the hidden layers (non-output) and all the Keras optimizers are all available > https://keras.io/optimizers/

**activation_out** = Same as 'activation' (above), but for the output layer only.

**save_model** =  An option to save the model configuration, weights and parameters.

OPTIONS:  default is 'False', if 'True' model will be saved with default name ('model') and if string, then the model name will be the string value e.g. 'titanic'.

**neuron_max** = The maximum number of neurons on any layer.

**neuron_last** = How many neurons there are in the last layer.

**batch_size** = Changes the number of samples that are propagated through the network at one given point in time. The smaller the batch_size, the longer the training will take.

**verbose** = This is set to '0' by default. The other options are '1' and '2' and will change the amount of information you are getting.

**shape** = Used for automatically creating a network shape. Currently there are 8 options available: 'funnel', 'rhombus', 'long_funnel', 'brick', 'hexagon', 'diamond', 'triangle', 'stairs'. Diagram is provided for each in the 'Shape' section. 

**double_check** = Makes a 'manual' check of the results provided by Keras backend and compares the two. This is good when you have doubt with the results.

**validation** = Validates in a more robust way than usual train/test split by initially splitting the dataset in half, where the first half becomes train and test, and then the second half becomes validation data set. 

OPTIONS: default is 'false', with 'true' 50% of data is separated for validation.

## Predictor

Once you've trained a model with train(), you can use it easily on any dataset through the predictor() command. You could use it in the Jypyter notebook, have it run on a server as part of some other process, or make it part of a website that does something interesting for the user based on their input. Just to name a few examples. Think of a trained neural net model as what is referred to as AI. It's far more easier to have AIs doing various tasks than most people think. Especially with Autonomio! 

```python
predictor(data,'model.json')
```
> Add labels to prections

```python
test(,data,labels='handle','model.json')
``` 

> Add an interactive scatter plot visualization with an y-axis variable::

```python
test(,data,'handle','model.json',y_scatter='influence_score')
``` 

> To yield the scatter plot, you have to call it specifically

```python
test_result = test('text',data,'handle','model.json',y_scatter='influence_score')
test_result[1]
``` 

### Test Query Parameters

ARGUMENT | REQUIRED INPUT | DEFAULT 
---------|----------------|--------
X | variable/s in dataframe | NA 
data | pandas dataframe | NA 
labels | variable/s in dataframe | NA
saved_model | filename | 5 
y_scatter | variable in dataframe | 'mean'


## Wrangler

## Data

The data() command is provided to allow data ingestion from a variety of formats, and to give the user access to unique deep learning datasets. In addition to allowing access to Autonomio datasets, the function also supports importing from csv, json, and excel. The data importing function is for most cases.

```python
# loading 'random_tweets' dataset in to a dataframe
df = data('random_tweets')

# loading data.csv in to a dataframe
df = data('data.csv',mode='file')
```

### Supported Formats

- csv 
- txt
- json
- msgpack (highly compressed binary format)

### Example datasets

Several unique deep learning focused datasets are provided with Autonomio. These datasets have not been released anywhere else, and relate to current affairs such as Twitter bots, ad fraud, US Election 2016, and party politics. 

- election_in_twitter
- programmatic_ad_fraund
- parties_and_employment
- tweet_sentiment
- random_tweets
- sites_category_and_vec


> Dataset consisting of 10 minute samples of 80 million tweets
    
```python
data('election_in_twitter')
```

> 4,000 ad funded websites with word vectors and 5 categories
   
```python
data('sites_category_and_vec')   
```

> Data from both buy and sell side and over 10 other sources
    
```python
data('programmatic_ad_fraud')    
```
    
> 9 years of monthly poll and unemployment numbers

```python
data('parties_and_employment')   
```
  
> 120,000 tweets with sentiment classification from NLTK
    
```python
data('tweet_sentiment')
```
    
> 20,000 random tweets

```python
data('random_tweets')            
```   

### Query Parameters 

ARGUMENT | REQUIRED INPUT | DEFAULT
---------|----------------|--------
name | dataset or filename | NA 
mode | string ('file') | 'default'
sep | string e.g '|' | ',' 
delimiter | string e.g ',' | None 
header | string ('file') | 'infer'

**name** = Name of the dataset or file. In the case of file, should be csv/txt for comma etc. separated values, json for json file and msgpack for msgpack. Automation of handling the request will not work unless the filename 

**mode** = Either 'default' which implies one of the Autonomio datasets, or 'file' which is for loading a file. 

**sep** =  By default ',' but can be any string. 

**delimiter** =  This is used as secondary for separator (sep). Should be string, for example ',' when thousand separators are used.

**header** =  Either integer for row number, 'None' for no header or default 'infer' will automatically decide (takes the top row mostly).

<aside class="success">
Most important thing to remember is to be nice and have fun! :) 
</aside>