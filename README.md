# Fuzzy-Rank-Ensemble
Based on our paper on "A Fuzzy Rank-based Ensemble of CNN Models for Classification of Cervical Cytology" under review in Nature- Scientific Reports.

# Requirements
To install the required dependencies run the following in command prompt:
`pip install -r requirements.txt`

# Running the codes:
To run the ensemble framework on cervical cytology data, download the data and follow the following directory structure:

```

+-- data
|   +-- .
|   +-- class1
|   +-- class2
|   +-- class3
+-- cnn_utils.py
+-- ensemble_utils.py
+-- main.py

```
Then, run the code using the command prompt as follows:
`python main.py --data_directory "data" --epochs 20`
