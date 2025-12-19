import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('sustainable_waste_management_dataset_2024.csv')

df.info()
df.head()

selected_features = ['recyclable_kg', 'waste_kg', 'collection_capacity_kg', 'temp_c']
X = df[selected_features]
y = df['waste_kg']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("MSE: ", mean_squared_error(Y_test, Y_pred))
print("R squared: ", r2_score(Y_test, Y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual Dollar Price (Y_test)')
plt.ylabel('Predicted Dollar Price (Y_pred)')
plt.title('Predicted vs. Actual Dollar Price')
plt.legend()
plt.grid(True)
plt.show()