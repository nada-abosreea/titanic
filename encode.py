import pandas as pd

def one_hot_encode(df):
	l_sex_dummies = pd.get_dummies(df['Sex'], drop_first=True)
	df1 = pd.concat([df,l_sex_dummies],axis=1)
	df.drop(['Sex'],axis=1,inplace=True)
	
	return df1