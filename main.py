# House Price Prediction - Simple ML Project

import numpy as np
import pandas as pd
import matplotllib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
  "Size_sqft": [500, 600, 750, 800, 900, 1000, 1200, 1500, 1800, 2000],
  "Price": [150000, 180000, 200000, 210000, 250000, 270000, 320000, 400000, 450000, 500000]
}
df = pd.Dataframe(data)
X = df[["Size_sqft]]
y = df["Price"]

