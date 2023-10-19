import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(0)
n_samples = 100
bedrooms = np.random.randint(1, 6, n_samples)
square_footage = np.random.randint(800, 2500, n_samples)
house_price = 100000 + 20000 * bedrooms + 150 * square_footage + np.random.normal(0, 10000, n_samples)

data = pd.DataFrame({'Bedrooms': bedrooms, 'SquareFootage': square_footage, 'Price': house_price})

X = data[['Bedrooms', 'SquareFootage']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

plt.scatter(X_test['SquareFootage'], y_test, label='True Prices', color='blue')
plt.scatter(X_test['SquareFootage'], y_pred, label='Predicted Prices', color='red')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.legend()
plt.show()
