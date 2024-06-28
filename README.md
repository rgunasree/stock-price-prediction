# Stock Price Prediction

This project predicts stock prices using machine learning models (Linear Regression, Random Forest, LightGBM) and a deep learning model (LSTM).

## Requirements

- pandas
- numpy
- matplotlib
- scikit-learn
- lightgbm
- tensorflow

## Setup

1. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib scikit-learn lightgbm tensorflow
2. Place your stock price data in an Excel file named 'stock_prices.xlsx' and ensure it has columns: Date, Open, High, Low, Close, AdjClose, Volume

## Usage

1. Load and Preprocess the Data
Load the dataset and preprocess it:
```bash
import pandas as pd

# Load the dataset
df = pd.read_excel('/content/stock_prices.xlsx')

# Preprocess the data
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
df['Date'] = pd.to_datetime(df['Date'])
subset_df = df.sample(frac=0.1, random_state=42)
subset_df['target'] = subset_df['Close'].shift(-1)
subset_df.drop(['Date'], axis=1, inplace=True)
subset_df.dropna(inplace=True)

# Define features and target
X = subset_df.drop('target', axis=1)
y = subset_df['target']

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
```

2. Train and Evaluate Machine Learning Models
Define the models and hyperparameters, perform grid search, and evaluate the models:
```
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from joblib import parallel_backend

# Define hyperparameter spaces and models
param_grid_linear = {'n_jobs': [-1]}
param_grid_rf = {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]}
param_grid_lgb = {'num_leaves': [31, 62, 127], 'learning_rate': [0.01, 0.1, 1]}
models = [(LinearRegression(), param_grid_linear), (RandomForestRegressor(), param_grid_rf), (lgb.LGBMRegressor(), param_grid_lgb)]

# Perform grid search
with parallel_backend('multiprocessing'):
    for model, param_grid in models:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"Best hyperparameters for {model.__class__.__name__}: {grid_search.best_params_}")
        print(f"Best score for {model.__class__.__name__}: {grid_search.best_score_}")
        print("")

# Train the best model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_valid)
print(f"RMSE on validation set: {mean_squared_error(y_valid, y_pred) ** 0.5}")

# Plotting the feature importances for the best model (assuming it's a tree-based model)
import matplotlib.pyplot as plt
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    plt.bar(range(len(importances)), importances)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.show()

```
3. Train and Evaluate LSTM Model
Preprocess the data, create, and train the LSTM model:

```
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Preprocess data for LSTM
df.set_index('Date', inplace=True)
data = df['Close'].values
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create and compile LSTM model
model = Sequential()
model.add(LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=256, return_sequences=True))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train LSTM model
early_stop = EarlyStopping(monitor='val_loss', patience=15)
history = model.fit(X_train, Y_train, epochs=200, batch_size=32, validation_data=(X_test, Y_test), callbacks=[early_stop])

# Evaluate LSTM model
loss = model.evaluate(X_test, Y_test, verbose=0)
print('Test Loss:', loss)

# Make predictions with LSTM model
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Plot results
plt.plot(Y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.legend()
plt.show()

# Save the model
model.save('stock_price_prediction_model.h5')

```

## Results

The model's performance is evaluated using RMSE (Root Mean Squared Error). Feature importance is plotted for tree-based models. LSTM model predictions are plotted against actual prices.


