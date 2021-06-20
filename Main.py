#Main

import pandas as pd
from sklearn.preprocessing import StandardScaler
import Model
import encode
import fill_nulls


sts=StandardScaler()
df1=pd.read_csv('../input/titanic/train.csv')
feature_scale=['Age','Fare']
df1[feature_scale]=sts.fit_transform(df1[feature_scale])

df1 = fill_nulls(df1)
df1 = one_hot_encode(df)
y = df1['label']
df1.drop('label',axis=1,inplace=True)
y_pred = Model(df1[0:200, :], y, df1[200:, :])