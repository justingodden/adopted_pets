# Will a pet be adopted

## Contents
The repo contains the following files:
* `data.py`: this contains functionality to retrieve task data, and split it into train/val/test datasets.
* `engineering.py`: this allows for feature engineering of the categorical variables.
* `modelling.py`: this provides all ML functionality - training, predicting, evaluating, saving, loading.
* `main.py`: this is the main script which ties all the parts together to complete the task.
* `model_dev.ipynb`: this Jupyter notebook contains the development of task 1 -- EDA, and model development -- before it was turned into python scripts. 
* artifacts: this contains the model output of the XGBoost Classifier.

## Running
To run the project, clone the repo, cd into the created folder and run the main file:
```bash
git clone https://github.com/justingodden/adopted_pets.git
cd adopted_pets
python main.py
```

An 'output' folder will be created with a results.csv file inside containing all the predictions generated with the trained XGBoost model.

The model file in the artifacts folder will also be overwritten with each running of the script.