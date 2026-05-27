# FSKT
Source code for FSKT

## Dataset
- [AS2009](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010)
- [AS2012](https://sites.google.com/site/assistmentsdata/2012-13-school-data-with-affect)
- [AS2017](https://sites.google.com/view/assistmentsdatamining/dataset)
- [AL2005](https://pslcdatashop.web.cmu.edu/KDDCup/)
- [BD2006](https://pslcdatashop.web.cmu.edu/KDDCup/)

## Preparation
- 1. Preprocess the raw datasets and generate the Q-matrix.
- 2. Multi-concept interactions should not be split; concepts for the same question are stored as a comma-separated string. Question and concept IDs should start from 1, with 0 reserved for padding.

The processed CSV file should include the following columns: `uid`, `fold`, `questions`, `concepts`, `responses`.
|uid|fold|questions|concepts|responses|
|--|--|--|--|--|
1|0|1|"5,4"|0
1|0|2|"4,3"|0
1|0|3|"3,1"|1
1|0|4|"2"|0
1|0|5|"1"|1


## Usage
- Step1: download the dataset, then put it in the folder `data/assist2009`.

- Step2: preprocess the dataset.

- Step3: Training.
```shell
python main.py --dataset assist2009
```

