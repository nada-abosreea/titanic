import pandas as pd

def fill_nulls(dataframe):
	dataframe['Age'].fillna(dataframe['Age'].mean(),inplace=True)
	dataframe['Fare'].fillna(dataframe['Fare'].mean(),inplace=True)
	
	return dataframe
	


def one_hot_encode(df):
	l_sex_dummies = pd.get_dummies(df['Sex'], drop_first=True)
	df1 = pd.concat([df,l_sex_dummies],axis=1)
	df.drop(['Sex'],axis=1,inplace=True)
	
	return df1
	
from sklearn.svm import SVC

def Model(x, y, a):
	model_svc=SVC(C=100,kernel='rbf')
	model_svc.fit(x, y)
	y_predicted=model_svc.predict(a)
	
	return y_predicted

	
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