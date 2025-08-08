import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

data = pd.read_csv(r"C:\Users\rac\Downloads\python machine learning model xgboost - electricity demand dataset.csv")

data

data.info()

data['Timestamp'] = pd.to_datetime(data['Timestamp'])

data.info()

data = data.set_index("Timestamp")

data

data[['Temperature', 'Humidity', 'Demand']].describe()

data.isnull().sum()

data[data.isna().any(axis=1)]

data[data.isna().all(axis=1)]

data = data.dropna(how = 'all')
data

data[data.isna().all(axis=1)]

data.isnull().sum()

data[['hour', 'dayofweek', 'month', 'year', 'dayofyear']] = data[['hour', 'dayofweek', 'month', 'year', 'dayofyear']].ffill()

data[['Temperature', 'Humidity']] = data[['Temperature', 'Humidity']].bfill()

data['Demand'] = data['Demand'].interpolate(method = 'time')

data.isnull().sum()

data.shape

data

data.insert(5, 'quarter', data.index.quarter)

data

data.info()

data[['hour', 'dayofweek', 'month', 'year', 'dayofyear']] = data[['hour', 'dayofweek', 'month', 'year', 'dayofyear']].astype(int)

data.info()

data

data.insert(5, 'weekofyear', data.index.isocalendar().week.astype(int))

data

data.tail(50)

data.insert(7, 'is_weekend', data.index.dayofweek.isin([5,6]))

data

data['is_weekend'] = data['is_weekend'].astype(int)

data

data[data['is_weekend'] == 1]

import holidays

data['Holidays'] = holidays.IN(years = data.year)

data

data.Holidays.value_counts()

data = data.drop('Holidays', axis=1)

data['Demand_lag_24hr'] = data['Demand'].shift(24)

data['demand_lag_168hr'] = data['Demand'].shift(168)

data.head(50)

data.iloc[160:200]

data['demand_rolling_mean_24hr'] = data['Demand'].rolling(window=24).mean()

data['demand_rolling_std_24hr'] = data['Demand'].rolling(window=24).std()

data.head(27)

data = data.dropna()
data

data['Demand'].plot(figsize=(15,6), title="Electricity Demand Over Time")
plt.xlabel("Year")
plt.ylabel("Demand (in MW)")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data = data, x='hour', y='Demand')
plt.title("Demand by Hour of the day");

plt.figure(figsize=(10,6))
sns.boxplot(data = data, x='month', y='Demand')
plt.title("Demand by Month")
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(data = data, x = 'Temperature', y= 'Demand', alpha = 0.5)
plt.title("Demand vs Temperature")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot= True, fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

data

Y = data.Demand

X = data.drop("Demand", axis=1)

print(Y)

X

X_train = X.loc[: '2048-12-31']

X_train

Y_train = Y.loc[: '2048-12-31']

Y_train

X_test = X.loc['2049-01-01':]

X_test

Y_test = Y.loc['2049-01-01':]

Y_test

print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

model_xgb = XGBRegressor(n_estimators=1000, early_stopping_rounds=50, learning_rate=0.01, random_state=42, objective='reg:squarederror')

model_xgb.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)], verbose = False)

predictions_xgb = model_xgb.predict(X_test)

rmse_xgb = np.sqrt(mean_squared_error(Y_test, predictions_xgb))

mae_xgb = mean_absolute_error(Y_test, predictions_xgb)

print('XGBoost RMSE:', rmse_xgb)
print('XGBoost MAE:', mae_xgb)

plt.figure(figsize=(15, 6))
plt.plot(Y_test.index, Y_test, label='Actual Demand', color = 'Blue')
plt.plot(Y_test.index, predictions_xgb, label='Predicted Demand', color= 'Red', linestyle='--')
plt.title('XGBoost Electricity Demand Prediction')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()

plt.show()

import joblib

joblib.dump(model_xgb,'electricity_xgb_predictions_model.pkl')

loaded_model = joblib.load('electricity_xgb_predictions_model.pkl')

future_predeictions = loaded_model.predict(X_future)

loaded_model