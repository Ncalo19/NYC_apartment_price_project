#Regression Example With Normal Dataset: Standardized and Large (more hidden layers)
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from numpy import loadtxt
from keras.optimizers import Adam

df = pd.read_csv(r'C:\Users\nCalo\Documents\Automifai\Research\Coding_Lessons\Git\NYC_apartment_price_project\Data\finished_data.csv')

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
'''
df.head()
dataset = df.values

# split into input (X) and output (Y) variablemodel.add(keras.layers.Dense(100, kernel_initializer='normal', activation='selu'))
X = dataset[:,0:391]
Y = dataset[:,391]
'''

# Seperate X (features) and Y (label)
X = df.drop(columns=['SALE PRICE'])
Y = df['SALE PRICE']

# build ml model with keras
model = keras.Sequential()
model.add(keras.layers.Dense(391, input_dim=391, kernel_initializer='normal', activation='relu')) #input dimensions must be == to number of features
model.add(keras.layers.Dense(150, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(50, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(20, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(10, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_absolute_error', optimizer=Adam(lr=.00005))
model.fit(X, Y, epochs=50, batch_size=100, verbose=2, shuffle=True) #epochs: how many times to run through, batch_size:how sets of data points to train on per epoch, verbose: how training progress is shown
model.save('NYC_apartment_price.h5') # save ml model
#https://www.youtube.com/watch?v=oCiRv94GMEc&feature=youtu.be&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw
# evaluate model with standardizestimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
