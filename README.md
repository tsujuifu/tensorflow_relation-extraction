# tensorflow_relation-extraction
A TensorFlow Implementation of Naive Relation Extraction under SemEval-2010 Task 8 (without entity specified)

## Requirements
+ **Python3**
+ TensorFlow >= 1.14
+ [Flair](https://github.com/zalandoresearch/flair)

## Usage
We use [Flair](https://github.com/zalandoresearch/flair) as **pre-trained embedding**

+ Data Processing
```
  python -m model_data-processing --emb=[glove, bert]
```

+ Train
```
  python -m model_relation-extraction --emb=[glove, bert]
```
