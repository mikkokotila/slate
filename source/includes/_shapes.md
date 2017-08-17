# Shapes

Shapes are used as part of the train() command, in order to dramatically change the network dimensions and shape with a single parameter. There are two parameters that work together to make up the shape and total neuron count of the neural network. 

- shape
- neuron_max 


> Examples: 

```python
# produce a long_funnel where the highest neuron per layer is 10 
train('text','neg',df,shape='long_funnel',neuron_max=10)

# produce a brick where the highest neuron per layer is 55 
train('text','neg',df,shape='brick',neuron_max=55)

```

NOTE: Shapes function is called from within the train() and does not serve a meaningful purpose for using separately. The function outputs a list with the neuron counts. 

Funnel
------

```
\          /
 \        /
  \      /
   \    /
    |  |
```

Funnel is the shape, which is set by default. It roughly looks like an upside-dowm pyramind, so that the first layer is defined as neuron_max, and the next layers are sligtly decreased compared to previous ones.

As funnel shape is set by default, we do not need to input anything to use it.

> Example input (default setting):

```python
tr = train(1,'neg',temp,layers=5,neuron_max=10)
```

For a five layer neural net, this will yield 10, 5, 3, 2, 1 neurons respectively. 


Long Funnel
-----------

```
 |          |
 |          |
 |          |
  \        /
   \      /
    \    /
     |  |
```

Long Funnel shape can be applied by defining shape as 'long_funnel'. First half of the layers have the value of neuron_max, and then they have the shape similar to Funnel shape - decreasing to the last layer.

> Example input:

```python
tr = train(1,'neg',temp,layers=5,neuron_max=10)
```

For a six layer neural net, this will yield 10, 10, 10, 5, 3, 2 neurons respectively. 


Rhombus
-------

```
     /   \
    /     \
   /       \
  /         \
  \         /
   \       /
    \     /
     \   /
     |   |
```

Rhobmus can be called by definind shape as 'rhombus'. The first layer equals to 1 and the next layers slightly increase till the middle one which equals to the value of neuron_max. Next layers are the previous ones goin in the reversed order. 

> Example input:

```python
train(1,'neg',temp,layers=5,neuron_max=10,shape='rhombus')
```

For a five layer neural net, this will yield 1, 6, 10, 6, 1 neurons respectively. 


Diamond
-------

```
   /       \
  /         \
  \         /
   \       /
    \     /
     \   /
     |   |
```

Defining shape as 'diamond' we will obtain the shape of the 'opened rhombus', where everything is similar to the Rhombus shape, but layers start from the larger number instead of 1. 


> Example input: 

```python
train(1,'neg',temp,layers=6,neuron_max=10,shape='diamond')
```

For a six layer neural net, this will yield 6, 6, 10, 5, 3, 2 neurons respectively. 


Hexagon
-------

```
    /    \
   /      \
  /        \
 |          |
 |          |
 |          |
  \        /
   \      /
    \    /
     |  |
```

Hexagon, which we get by calling 'hexagon' for shape, starts with 1 as the first layer and increases till the neuron_max value. Then some next layers will have maximum value untill it starts to decrease till the last layer. 

> Example input:

```python
train(1,'neg',temp,layers=7,neuron_max=10,shape='hexagon')
```

Output list of neurons(excluding ounput layer). 


For a seven layer neural net, this will yield 1, 3, 5, 10, 10, 5, 3 neurons respectively.


Brick
-----

```
   |             |
   |             |
   |             |
   |             |
    ----     ----
        |   |

```

All the layers have neuron_max value. Called by shape='brick'. 

> Example input:

```python
    tr = train(1,'neg',temp,layers=5,neuron_max=10,shape='brick')
```

Output list of neurons(excluding ounput layer). 

For a five layer neural net, this will yield 10, 10, 10, 10, 10 neurons respectively.


Triangle
--------

```
        /    \
       /      \
      /        \
     /          \
    /            \
    ----      ----
        |    |
```
This shape, which is called by defining shape as 'triangle' starts with 1 and increases till the last input layer, which is neuron_max. 

> Example input: 


```python
train(1,'neg',temp,layers=5,neuron_max=10,shape='triangle')
```

Output list of neurons(excluding ounput layer). 


For a five layer neural net, this will yield 1, 2, 3, 5, 10 neurons respectively.


Stairs
------

```
   |                      |
    ---                ---
       |             |
        ---       ---
           |     |
```

You can apply it defining shape as 'stairs'. If number of layers more than four, then each two layers will have the same value, then it decreases.If the number of layers is smaller than four, then the value decreases every single layer. 

> Example input: 

```python
train(1,'neg',temp,layers=6,neuron_max=10,shape='stairs')
```

For a six layer neural net, this will yield 10, 10, 8, 8, 6, 6 neurons respectively.

