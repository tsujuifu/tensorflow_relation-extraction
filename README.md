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
  python -m main_data-processing --emb=[glove, bert]
```

+ Train
```
  python -m main_relation-extraction --emb=[glove, bert]
```

## Result
| Embedding | Training Loss | Validation Accuracy | Testing Accuracy |
| :-: | :-: | :-: | :-: | 
| GloVe | 1.3609 | 53.37% | 56.815% | 
| BERT | 0.4248 | 69.12% | 69.519% |

## Resources
+ [SemEval-2010 Task 8](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view)
