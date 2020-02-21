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

@app.route('/', methods=['POST']) # https://github.com/nitinkaushik01/Deploy_Machine_Learning_Model_on_Flask_App/blob/master/Flask_Sample_App/app.py
def predict():
    # grabs the input data when a post request is sent
    ResU= int(request.form['Residential_Units'])
    ComU= int(request.form['Commercial_Units'])
    Lsqft= float(request.form['Land_sqft'])
    Gsqft= float(request.form['Gross_sqft'])
    Neighb= str(request.form['Neighborhood'])
    Class_category= str(request.form['Building_Class_Category'])
    Class= str(request.form['Building_Class'])
    Tax= str(request.form['Tax_Class'])
    Year= (request.form['Year_Built'])

    #convert year to the str category that the model will understand
    Year = int(Year)
    if Year < 3:
        Year ='AGE OF BUILDING_1'
    elif Year < 10:
        Year = 'AGE OF BUILDING_2'
    elif Year < 20:
        Year = 'AGE OF BUILDING_3'
    elif Year < 30:
        Year = 'AGE OF BUILDING_4'
    elif Year < 50:
        Year = 'AGE OF BUILDING_5'
    elif Year < 75:
        Year = 'AGE OF BUILDING_6'
    elif Year < 100:
        Year = 'AGE OF BUILDING_7'
    elif Year < 150:
        Year = 'AGE OF BUILDING_8'
    else:
        Year = 'AGE OF BUILDING_9'

    # cleans the dataframe using pandas (dataframe needed here because it is used to create the input columns at line 84)
    import datetime
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df = pd.read_csv('https://raw.githubusercontent.com/Ncalo19/NYC_sales_data/master/2_cleaned_NYC_property_sales.csv')
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
    df=df.rename(columns=lambda x: x.strip())
    df.columns = [col.replace('NEIGHBORHOOD_', '') for col in df.columns]
    df.columns = [col.replace('BUILDING CLASS CATEGORY_', '') for col in df.columns]
    df.columns = [col.replace('BUILDING CLASS AT TIME OF SALE_', '') for col in df.columns]
    df.columns = [col.replace('TAX CLASS AT TIME OF SALE_', '') for col in df.columns]

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
