import pandas as pd

def fill_nulls(dataframe):
	dataframe['Age'].fillna(dataframe['Age'].mean(),inplace=True)
	dataframe['Fare'].fillna(dataframe['Fare'].mean(),inplace=True)
	print("ok")
	return dataframe
      
