# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:05:34 2022

@author: andyh
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('Walmart/wmart_data_compiled.csv', parse_dates=['Date'])
df['Date'] = pd.to_numeric(df['Date'])

df.loc[df.MarkDown1.isnull(), 'MarkDown1'] = 0
df.loc[df.MarkDown2.isnull(), 'MarkDown2'] = 0
df.loc[df.MarkDown3.isnull(), 'MarkDown3'] = 0
df.loc[df.MarkDown4.isnull(), 'MarkDown4'] = 0
df.loc[df.MarkDown5.isnull(), 'MarkDown5'] = 0



le = preprocessing.LabelEncoder()
le.fit(df['Type'].unique())
df['Type'] = le.transform(df['Type'])


def WMAE(ds, expt, pred):
    w = ds.IsHoliday.apply(lambda x: 5 if x else 1)
    return np.sum(w*abs(pred-expt)/np.sum(w))
    


y = df['Weekly_Sales']
X = df.drop(['Weekly_Sales'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(f'MAE for Linear Regression - {mean_absolute_error(y_test,y_pred)}')
print(f'WMAE for linear Regression - {WMAE(X_test, y_test, y_pred)}')
      
      
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f'MAE for Random Forest - {mean_absolute_error(y_test,y_pred)}')
print(f'WMAE for Random Forest - {WMAE(X_test, y_test, y_pred)}')

importance = rf.feature_importances_
      
index_top_10 = np.argsort(importance)[::-1][:10]
columns_top_10 = [X_train.columns[i] for i in index_top_10]
      
plt.bar(range(10), importance[index_top_10])
plt.xticks(range(10), columns_top_10)      
