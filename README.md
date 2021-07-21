[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-fuzzy-rank-based-ensemble-of-cnn-models-for/image-classification-on-sipakmed)](https://paperswithcode.com/sota/image-classification-on-sipakmed?p=a-fuzzy-rank-based-ensemble-of-cnn-models-for)
# Fuzzy-Rank-Ensemble
Based on our paper ["A Fuzzy Rank-based Ensemble of CNN Models for Classification of Cervical Cytology"](https://www.nature.com/articles/s41598-021-93783-8#article-info) published in Nature- Scientific Reports.

<img src="/overall.png" style="margin: 10px;">

# Requirements
To install the required dependencies run the following in command prompt:
`pip install -r requirements.txt`

# Running the codes:
To run the ensemble framework on cervical cytology data, download the data and follow the following directory structure:

(Note: Any number of classes can be present in the dataset. It will be captured by the code automatically)

```

+-- data
|   +-- .
|   +-- class1
|   +-- class2
|   +-- class3
|   +-- class4
+-- utils
|   +-- .
|   +-- utils_cnn
|   +-- utils_ensemble
+-- main.py

```
Then, run the code (5-fold cross-validation will be automatically performed) using the command prompt as follows:

`python main.py --data_directory "data/" --epochs 20`

## Citation

If this repository helps you in any way, consider citing our paper as follows:
```
@article{manna2021fuzzy,
  title={A fuzzy rank-based ensemble of CNN models for classification of cervical cytology},
  author={Manna, Ankur and Kundu, Rohit and Kaplun, Dmitrii and Sinitca, Aleksandr and Sarkar, Ram},
  journal={Scientific Reports},
  volume={11},
  number={1},
  pages={1--18},
  year={2021},
  publisher={Nature Publishing Group}
}
```
