import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle

with open("winning_model.pkl", "rb") as f:
        lawnmower_model = pickle.load(f)
        
print("\n*******************")
print("* USF Riding Mower Classification *")
print("*******************\n")

inc = float(input('Enter the income (in thousands of dollars): '))
lot_size = float(input('Enter the lot size (in thousands of square feet): '))

df = pd.DataFrame({'Income': [inc], 'LotSize': [lot_size]})

class_prediction = lawnmower_model.predict(df)
probability = lawnmower_model.predict_proba(df)

ownership = ('Nonowner', 'Owner')
print(probability)
print(f"\nUSF Riding Mower Classifier indicates that probability of ownership at {probability[0][1]:.4f}, therefore it's indicated that the person is {class_prediction[0]}.\n")