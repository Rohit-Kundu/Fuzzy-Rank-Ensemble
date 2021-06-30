# Fuzzy-Rank-Ensemble
Based on our paper on "A Fuzzy Rank-based Ensemble of CNN Models for Classification of Cervical Cytology" accepted for publication in Nature- Scientific Reports.

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
+-- cnn_utils.py
+-- ensemble_utils.py
+-- main.py

```
Then, run the code (5-fold cross-validation will be automatically performed) using the command prompt as follows:

`python main.py --data_directory "data/" --epochs 20`
