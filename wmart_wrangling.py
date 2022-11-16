# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:16:46 2022

@author: andyh
"""

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
import datetime as dt


df = pd.read_csv('Walmart/wmart_data_compiled.csv')

df['IsHoliday'] = df['IsHoliday'].apply(lambda x: 1 if x==True else 0)
le = preprocessing.LabelEncoder()
le.fit(df['Type'].unique())


        
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Week'] = df['Date'].dt.week
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day



df['Holiday_Type'] = df['Week'].apply(lambda x: 0 if x not in [6,36,47,52] else x)
le = preprocessing.LabelEncoder()

x = df[df['IsHoliday']==1]['Week'].unique()
x = x.tolist()
x.append(0)
le.fit(x)

df['Holiday_Type'] = le.transform(df['Holiday_Type'])
print(df['Holiday_Type'].unique())

cor = df[['Holiday_Type', 'IsHoliday', 'Weekly_Sales']].corr()
sns.heatmap(cor, annot=True, cmap="crest")
plt.show()

#result shows holiday type and isholiday both only have 0.013 correlation
#this emans holiday/no holiday and the types of holiday has low correlation

def weeks_pre_holiday(x):
    diff_list=[]
    if x['Year'] == 2010:
        
        for d in [dt.datetime(2010, 12, 31),
                  dt.datetime(2010, 11, 26),
                  dt.datetime(2010, 9, 10),
                  dt.datetime(2010, 2, 12)]:
            
            d_diff = d - x['Date']
            if d_diff.days < 0:
                diff_list.append(0)
            else:
                diff_list.append(d_diff.days / 7)
            return int(min(diff_list))  
    
    if x['Year'] == 2011:
        
        for d in [dt.datetime(2011, 12, 30),
                  dt.datetime(2011, 11, 25),
                  dt.datetime(2011, 9, 9),
                  dt.datetime(2011, 2, 11)]:
            
            d_diff = d - x['Date']
            if d_diff.days < 0:
                diff_list.append(0)
            else:
                diff_list.append(d_diff.days / 7)  
                
            return int(min(diff_list))

    if x['Year'] == 2012:
        
        for d in [dt.datetime(2012, 12, 28),
                  dt.datetime(2012, 11, 23),
                  dt.datetime(2012, 9, 7),
                  dt.datetime(2012, 2, 10)]:
            
            d_diff = d - x['Date']
            if d_diff.days < 0:
                diff_list.append(0)
            else:
                diff_list.append(d_diff.days / 7)
            return int(min(diff_list))           

df['weeks_pre_holiday'] = df.apply(weeks_pre_holiday, axis=1)

print(df['weeks_pre_holiday'].head(100))

