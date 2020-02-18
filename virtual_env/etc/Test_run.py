import numpy as np
from numpy import loadtxt
from keras.models import load_model
from pandas import read_csv
import datetime
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

X = df.drop(columns=['SALE PRICE'])
Y = df['SALE PRICE']


model = keras.Sequential()
model.add(keras.layers.Dense(422, input_dim=422, kernel_initializer='normal', activation='selu'))
model.add(keras.layers.Dense(100, kernel_initializer='normal', activation='selu'))
model.add(keras.layers.Dense(5, kernel_initializer='normal', activation='selu'))
model.add(keras.layers.Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
model.fit(X, Y, epochs=10, batch_size=150, verbose=2, shuffle=True)
model.save('NYC_apartment_price.h5')

def predict_price(Residential_Units, Commercial_Units, Land_sqft, Gross_sqft, Neighborhood, Building_Class_Category, Building_Class, Tax_Class, Year_Built):
    neighborhood_index= np.where(X.columns==Neighborhood)[0][0]
    Building_Class_Category_index= np.where(X.columns==Building_Class_Category)[0][0]
    Building_Class_index= np.where(X.columns==Building_Class)[0][0]
    tax_index= np.where(X.columns==Tax_Class)[0][0]
    year_index= np.where(X.columns==Year_Built)[0][0]

    x=np.zeros(len(X.columns))
    x[0]= Residential_Units
    x[1]= Commercial_Units
    x[2]= Land_sqft
    x[3]= Gross_sqft
    if neighborhood_index >= 0:
        x[neighborhood_index] = 1
    if Building_Class_Category_index >= 0:
        x[Building_Class_Category_index] = 1
    if Building_Class_index >= 0:
        x[Building_Class_index] = 1
    if tax_index >= 0:
        x[tax_index] = 1
    if year_index >= 0:
        x[year_index] = 1

    #return model.predict([x])[0]
    test1 = np.array([x])[0]
    return model.predict(test1.reshape(1, 422), batch_size=1)

ResU= 3
ComU= 0
Lsqft= 2000
Gsqft= 3949
Neighb= 'NEIGHBORHOOD_BEDFORD STUYVESANT'
Class_category= 'BUILDING CLASS CATEGORY_29 COMMERCIAL GARAGES                      '
Class= 'BUILDING CLASS AT TIME OF SALE_H6'
Tax= 'TAX CLASS AT TIME OF SALE_1'
Year= 'AGE OF BUILDING_2'

'''
The below method does not work because it gets mixed up with the global and local versions of the below variables
Residential_Units= 3
Commercial_Units= 0
Land_sqft= 2000
Gross_sqft= 3949
Neighborhood= 'NEIGHBORHOOD_BEDFORD STUYVESANT'
Building_Class_Category= 'BUILDING CLASS CATEGORY_29 COMMERCIAL GARAGES                      '
Building_Class= 'BUILDING CLASS AT TIME OF SALE_H6'
Tax_Class= 'TAX CLASS AT TIME OF SALE_1'
Year_Built= 'AGE OF BUILDING_2'
'''
#Year_Built=(datetime.datetime.now().year)-Year_Built

prediction = predict_price(ResU,ComU,Lsqft,Gsqft,Neighb,Class_category,Class,Tax,Year)
print(prediction)

#print(predict_price(3, 0, 3000, 3949, 'NEIGHBORHOOD_BEDFORD STUYVESANT', 'BUILDING CLASS CATEGORY_29 COMMERCIAL GARAGES                      ', 'BUILDING CLASS AT TIME OF SALE_H6', 'TAX CLASS AT TIME OF SALE_1', 'AGE OF BUILDING_2'))
