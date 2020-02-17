#Regression Example With Normal Dataset: Standardized and Large (more hidden layers)
import pandas as pd
import numpy as np
import keras
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from numpy import loadtxt

import datetime
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df = pd.read_csv(r'C:\Users\nCalo\Documents\Automifai\Research\Coding_Lessons\Git\NYC_apartment_price_project\Data\2_cleaned_NYC_property_sales.csv')
df = pd.get_dummies(df, columns=['NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'BUILDING CLASS AT TIME OF SALE', 'TAX CLASS AT TIME OF SALE'])
df['current_year'] = datetime.datetime.now().year
df['YEAR BUILT'].astype(int)
df['AGE OF BUILDING'] = df['current_year']-df['YEAR BUILT']
bins = [0,3,10,20,30,50,75,100,150,1000]
labels = [1,2,3,4,5,6,7,8,9]
df['AGE OF BUILDING'] = pd.cut(df['AGE OF BUILDING'], bins=bins, labels=labels, right=True)
df['AGE OF BUILDING']=df['AGE OF BUILDING'].astype('object')
df = pd.get_dummies(df, columns=['AGE OF BUILDING'])
df = df.drop(columns=['#','BOROUGH','BLOCK', 'YEAR BUILT', 'current_year', 'TOTAL UNITS'])
df_SALEPRICE = df.pop('SALE PRICE')
df['SALE PRICE']=df_SALEPRICE

'''
X = df.drop('SALE PRICE',axis='columns')
print(X.head(10))
y=df['SALE PRICE']
print(y.head(10))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
print(lr_clf.score(X_test,y_test))

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

print(cross_val_score(LinearRegression(), X, y, cv=cv))
'''
df.head()
dataset = df.values

# split into input (X) and output (Y) variables
X = dataset[:,0:422]
Y = dataset[:,422]

model = keras.Sequential()
model.add(keras.layers.Dense(422, input_dim=422, kernel_initializer='normal', activation='selu'))
model.add(keras.layers.Dense(100, kernel_initializer='normal', activation='selu'))
model.add(keras.layers.Dense(5, kernel_initializer='normal', activation='selu'))
model.add(keras.layers.Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=100, batch_size=150, verbose=2, shuffle=True)
model.save('NYC_apartment_price.h5')



# evaluate model with standardizestimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
