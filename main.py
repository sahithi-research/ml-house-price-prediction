# House Price Prediction - Simple ML Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

X = df[["Size_sqft"]]
y = df["Price"]

# Train test split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Model training

model = LinearRegression()
model.fit(X_train,y_train)

# Prediction

y_pred = model.predict(X_test)

# Evaluation

print("Mean Squared Error:", mean_squared_error(y_test,y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization

plt.scatter(X, y, color = "blue", label = "Actual Data")
plt.plot(X, model.predict(X), color = "red", label = "Prediction Line")
plt.xlabel("House Size (sqft)")
plt.ylabel("Prices")
plt.title("House Price Prediction using Linear Regression")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

# predict new house price

new_size = np.array([[1100]])
predicted_price = model.predict(new_size)
print("Predicted price for 1100 sqft:", predicted_price[0])

