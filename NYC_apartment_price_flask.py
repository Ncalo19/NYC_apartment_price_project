# flask.py
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import url_for
import numpy as np
from numpy import loadtxt
from keras.models import load_model
import pandas as pd
from pandas import read_csv
import datetime


app = Flask(__name__)

# load model
model = load_model('NYC_apartment_price.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) # https://github.com/nitinkaushik01/Deploy_Machine_Learning_Model_on_Flask_App/blob/master/Flask_Sample_App/app.py
def predict():
    # grabs the input data when a post request is sent
    ResU= int(request.form['Residential_Units'])
    ComU= int(request.form['Commercial_Units'])
    Lsqft= float(request.form['Land_sqft'])
    Gsqft= float(request.form['Gross_sqft'])
    Neighb= (request.form['Neighborhood'])
    Class_category= (request.form['Building_Class_Category'])
    Class= (request.form['Building_Class'])
    Tax= (request.form['Tax_Class'])
    Year= (request.form['Year_Built'])
    #Year_Built=(datetime.datetime.now().year)-Year_Built

    # cleans the dataframe using pandas
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

    # divides the data set into X (data) and Y (desired predicted value)
    X = df.drop(columns=['SALE PRICE'])
    Y = df['SALE PRICE']

    def predict_price(Residential_Units, Commercial_Units, Land_sqft, Gross_sqft, Neighborhood, Building_Class_Category, Building_Class, Tax_Class, Year_Built):
        neighborhood_index= np.where(X.columns==Neighborhood)[0][0] # finds column with the title given in the neighborhood box
        Building_Class_Category_index= np.where(X.columns==Building_Class_Category)[0][0]
        Building_Class_index= np.where(X.columns==Building_Class)[0][0]
        tax_index= np.where(X.columns==Tax_Class)[0][0]
        year_index= np.where(X.columns==Year_Built)[0][0]

        x=np.zeros(len(X.columns)) #sets all columns in a data set object x to zero
        x[0]= Residential_Units #changes a specified column from zero to the value assigned to the variable (retreived from the post request)
        x[1]= Commercial_Units
        x[2]= Land_sqft
        x[3]= Gross_sqft
        if neighborhood_index >= 0:
            x[neighborhood_index] = 1 # assigns a one to the desired neighborhood (one hot encoding)
        if Building_Class_Category_index >= 0:
            x[Building_Class_Category_index] = 1
        if Building_Class_index >= 0:
            x[Building_Class_index] = 1
        if tax_index >= 0:
            x[tax_index] = 1
        if year_index >= 0:
            x[year_index] = 1

        #return model.predict([x])[0]
        test1 = np.array([x])[0] #the x data set object is passed through the ml model
        return model.predict(test1.reshape(1, 422), batch_size=1)

    prediction = predict_price(ResU,ComU,Lsqft,Gsqft,Neighb,Class_category,Class,Tax,Year) # set up this way to avoid confusion between global and local variables
    return render_template('index.html', prediction_text='Price should be {}'.format(prediction)) # prediction sent to index.html template file

if __name__ == "__main__":
    app.run(debug=True)
