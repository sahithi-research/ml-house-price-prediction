# House Price Prediction - Simple ML Project

import numpy as np
import pandas as pd
import matplotllib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset

data = {
  "Size_sqft": [500, 600, 750, 800, 900, 1000, 1200, 1500, 1800, 2000],
  "Price": [150000, 180000, 200000, 210000, 250000, 270000, 320000, 400000, 450000, 500000]
}
df = pd.Dataframe(data)

# Features (X) and target (y)

X = df[["Size_sqft]]
y = df["Price"]

# Train test split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Model training

model = LinearRegression()
model.fit(X_train,y_train)

