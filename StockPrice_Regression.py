# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 14:50:46 2019

@author: Ram Prabodh Induri
"""

import quandl
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math,datetime
from matplotlib import style
style.use('ggplot') #to make plots look decent
df = quandl.get('WIKI/GOOGL')

df= df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

df['HL_PCT']= (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] *100
df['PCT_CNG']= (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] *100

df=df[['Adj. Close','HL_PCT','Adj. Volume','PCT_CNG']]

forecast_column ='Adj. Close'

df.fillna(-10000,inplace=True) #replacing missing values with -10,000
forecast_out = int(math.ceil(0.005*len(df))) #forecast_out defines the number of days shifted into future.

print(" The label is Adj. Close" , forecast_out, "days into the future")
df['label'] = df[forecast_column].shift(-forecast_out)#the label is Adj.Close some days into future determined by forecast_out

 #dropping the columns with no labels ("columns at the end")

X=np.array(df.drop(['label'],1)) #df.drop() returns a new dataframe so 'label' column exists in original df dataframe
X=preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]


df.dropna(inplace=True)
y=np.array(df['label'])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15)

cla= LinearRegression(n_jobs=-1) #n_jobs is used to Thread and do multiple computations at once; -1 implies maximum possible .
cla.fit(X_train,y_train)
print("Accuracy is : \n")
accuracy = cla.score(X_test,y_test)
print(accuracy)
print("                                                           \n                                              \n")
#print(forecast_out) # so we are shifting by 18 days as forecast_out turned out to be 18 in my case
forecast_set=cla.predict(X_lately)
print("Forecatsed/missing values, predicted are : \n")
print(forecast_set,accuracy,forecast_out)

df['Forecast'] = np.nan

last_date=df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set: #To have dates on the axis for plotting.
    next_date= datetime.datetime.fromtimestamp(next_unix)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

