# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:29:36 2022

@author: andyh
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns


df = pd.read_csv('Walmart/wmart_data_compiled.csv')
df['IsHoliday'] = df['IsHoliday'].apply(lambda x: 1 if x==True else 0)


le = preprocessing.LabelEncoder()
le.fit(df['Type'].unique())
df['Type'] = le.transform(df['Type'])
             
sns.jointplot(data=df, x = "Temperature", y="Weekly_Sales")
# plt.show()

t_30_40 = df[((df['Temperature']>30) & (df['Temperature']<40 ))].Weekly_Sales.sum()
t_40_50 = df[((df['Temperature']>40) & (df['Temperature']<50 ))].Weekly_Sales.sum()
t_50_60 = df[((df['Temperature']>50) & (df['Temperature']<60 ))].Weekly_Sales.sum()
t_60_70 = df[((df['Temperature']>60) & (df['Temperature']<70 ))].Weekly_Sales.sum()
t_70_80 = df[((df['Temperature']>70) & (df['Temperature']<80 ))].Weekly_Sales.sum()
t_80_90 = df[((df['Temperature']>80) & (df['Temperature']<90 ))].Weekly_Sales.sum()
t_90_100 = df[((df['Temperature']>90) & (df['Temperature']<100 ))].Weekly_Sales.sum()

t_list = [t_30_40,t_40_50,t_50_60,t_60_70,t_70_80,t_80_90,t_90_100]
t_df = pd.Series(t_list, index=['t_30_40','t_40_50','t_50_60','t_60_70','t_70_80','t_80_90','t_90_100'])
t_df.index.name = 'temp'
t_df.plot()
plt.show()

t_df = t_df.sort_values(ascending=True)
t_df = t_df.to_frame()
t_df.reset_index(inplace=True)

