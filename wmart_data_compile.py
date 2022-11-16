import pandas as pd


path = "" #path of the original threee Walmart Data files

df_t = pd.read_csv(path+r"\train.csv", parse_dates=['Date'])
df_f = pd.read_csv(path+r"\features.csv", parse_dates=['Date'])
df_s = pd.read_csv(path+r"\stores.csv")


df_tf = df_t.merge(df_f, on=['Store', 'Date', 'IsHoliday'], how='inner' )
df_tfs = df_tf.merge(df_s, on=['Store'], how='inner')

df_tfs.to_csv('Walmart\wmart_data_compiled.csv')
print(df_tf)
print(df_tfs.shape)
print(df_tfs.describe())

print(df_tfs.MarkDown1.unique())

df_tfs.loc[df_tfs.MarkDown1.isnull(), 'MarkDown1'] = 0
df_tfs.loc[df_tfs.MarkDown2.isnull(), 'MarkDown2'] = 0
df_tfs.loc[df_tfs.MarkDown3.isnull(), 'MarkDown3'] = 0
df_tfs.loc[df_tfs.MarkDown4.isnull(), 'MarkDown4'] = 0
df_tfs.loc[df_tfs.MarkDown5.isnull(), 'MarkDown5'] = 0
print(df_tfs.isnull().sum())

print(df_tfs.groupby('Date')['Weekly_Sales'].sum())

#df_tfs.groupby('Date')['Weekly_Sales'].sum().plot()

import seaborn as sns
sns.set_style("whitegrid")
s = sns.distplot(df_tfs.groupby('Date')['Weekly_Sales'].sum()).set()
s.set(title='Weekly Store Sales Distribution')

#most weekly sales are around 40 million



sns.distplot(df_tfs.groupby('Date')['MarkDown1'].sum())
sns.distplot(df_tfs.groupby('Date')['MarkDown2'].sum())
sns.distplot(df_tfs.groupby('Date')['MarkDown3'].sum())
sns.distplot(df_tfs.groupby('Date')['MarkDown4'].sum())
sns.distplot(df_tfs.groupby('Date')['MarkDown5'].sum())

sns.countplot(df_tfs.Type)

sns.distplot(df_tfs[df_tfs.Type=='A'].groupby('Date')['Weekly_Sales'].sum())
sns.distplot(df_tfs[df_tfs.Type=='B'].groupby('Date')['Weekly_Sales'].sum())
sns.distplot(df_tfs[df_tfs.Type=='C'].groupby('Date')['Weekly_Sales'].sum())


df_tfs[df_tfs.Type=='A'].groupby('Date')['Weekly_Sales'].sum().plot()
df_tfs[df_tfs.Type=='B'].groupby('Date')['Weekly_Sales'].sum().plot()
df_tfs[df_tfs.Type=='C'].groupby('Date')['Weekly_Sales'].sum().plot()
# holiday has no effect on C type store
