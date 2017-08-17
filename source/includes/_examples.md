# Examples

Autonomio is very easy to use and it's straightforward to memorize the namespace which is just 4 commands and less than 40 arguments combined. Namespace memorization is one of the key differences between advanced and beginner users. Whereas Autonomio helps lower skill level practitioners to dramatically improve their capability, advanced practitioners enjoy significant productivity gains and headache reduction.

## Prepare and Train

A typical use-case, even with messy datasets with many columns, involves few lines of code and seconds or minutes of training time on a regular laptop machine.

> Medicare Provider Utilization and Payment Data

```python

# do the python imports 
from autonomio.commands import data, wrangler, train, predictor
%matplotlib inline

# import the data from csv
df = data('medicare_10k.csv', mode='file', header=None)

df = wrangler(df,'z')

# train a neural net
train([2,17],'z',df,epoch=20,loss='logcosh',flatten='median')
```



