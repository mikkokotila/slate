# Language Processing

## Unstructed Data

By some estimates, more than 90% of meaningful data is unstructured. Ingestion of unstructured data with Autonomio could not be easier; inputting unstructured data as 'x' is handled automatically whereas the input is converted in to word2vec word vectors. The way this works is roughly:

1) detect if a single column of x features is text
2) use spaCy NLP to vectorize the text 
3) create 300 invididual features/columns from the vector
4) use the 300 features as signals for training the model 

In addition to doing this automatically with train() having a single x column with text, when one or more columns of text needs to be vectorized as part of a dataset with other features, this can be done easily by using the 'vectorize' parameter in train().

Also the wrangler() data preparation function can be used to vectorize unstructured features (e.g. tweets or names).

## Language support

Autonomio's vectorizing engine spaCy supports currently 13 languages: 

- English
- German
- Chinese
- Spanish
- Italian
- French
- Portuguese
- Dutch
- Swedish
- Finnish
- Hungarian
- Bengali
- Hebrew

NOTE: the spacy language libraries have to be downloaded each separately.

[Read spaCy's language page](https://spacy.io/docs/api/language-models)

## Adding new languages 

spaCy makes it reletively streamlined to create support for any language and the challenge can (and should be) approached iteratively. 