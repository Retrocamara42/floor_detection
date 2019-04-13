# Analizing the data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./input/X_train.csv")

serie1 = df[df['series_id']==0]
print(df.head())
#print(df)
#serie1.plot.scatter(x='orientation_X', y='orientation_Y')
#plt.show()
print(df.series_id.unique())

print(len(df))

df2 = pd.read_csv("./input/y_train.csv")

print(df2.head())
print(df2.group_id.unique())
print(df2[df2['series_id']<=2660].series_id.unique())
print(df2.surface.unique())


#df2[df2['group_id']==13]['surface'].value_counts().plot.bar()
#plt.show()
'''
f,ax = plt.subplots(figsize=(6,6))
m = df.iloc[:,3:].corr()
sns.heatmap(m, annot=True, linecolor='darkblue', linewidths=.1, cmap="YlGnBu", fmt= '.1f',ax=ax)
plt.show()
'''
