import pandas as pd

import modelling


def test_predict():
    test_dict = {'Type': 0,
                 'Age': 12,
                 'Breed1': 5,
                 'Gender': 1,
                 'MaturitySize': 1,
                 'FurLength': 1,
                 'Vaccinated': -1,
                 'Sterilized': -1,
                 'Health': 0,
                 'Fee': 250,
                 'PhotoAmt': 3,
                 'Color1_Black': 1,
                 'Color1_Brown': 0,
                 'Color1_Cream': 0,
                 'Color1_Golden': 0,
                 'Color1_Gray': 0,
                 'Color1_White': 0,
                 'Color1_Yellow': 0,
                 'Color2_Brown': 0,
                 'Color2_Cream': 0,
                 'Color2_Golden': 0,
                 'Color2_Gray': 0,
                 'Color2_No Color': 0,
                 'Color2_White': 1,
                 'Color2_Yellow': 0}

    test_data = pd.DataFrame(test_dict, index=[0])
    modeller = modelling.Modeller()
    modeller.load()
    assert modeller.predict(test_data) == 1
