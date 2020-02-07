# flask.py
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import url_for
import numpy as np
from numpy import loadtxt
from keras.models import load_model
from pandas import read_csv

app = Flask(__name__)

# load model
model = load_model('NYC_apartment_price.h5')
# summarize model.z
model.summary()
# load dataset
dataframe = read_csv(r'C:\Users\nCalo\Documents\Automifai\Research\Coding_Lessons\Git\NYC_apartment_price_project\Data\2_cleaned_NYC_property_sales.csv', delimiter=",")
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:12]
Y = dataset[:,12]
# evaluate the model


@app.route('/')
def home():
    return render_template('html.html')

@app.route('/predict', methods=['POST']) # https://github.com/nitinkaushik01/Deploy_Machine_Learning_Model_on_Flask_App/blob/master/Flask_Sample_App/app.py
def predict():
    Borough= float(request.form['Borough'])
    Block= float(request.form['Block'])
    Neighborhood= float(request.form['Neighborhood'])
    Building_Class_Category= float(request.form['Building_Class_Category'])
    Residential_Units= float(request.form['Residential_Units'])
    Commercial_Units= float(request.form['Commercial_Units'])
    Total_Units= float(request.form['Total_Units'])
    Land_sqft= float(request.form['Land_sqft'])
    Gross_sqft= float(request.form['Gross_sqft'])
    Year_Built= float(request.form['Year_Built'])
    Building_Class= float(request.form['Building_Class'])
    Tax_Class= float(request.form['Tax_Class'])


    prediction = model.predict(features.reshape(1, 12), batch_size=1)

    return render_template('html.html', prediction_text='Price should be {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
