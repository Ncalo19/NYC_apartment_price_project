import pandas as pd
from pandas import read_csv
import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np

le = LabelEncoder()
df = pd.read_csv('https://raw.githubusercontent.com/Ncalo19/NYC_sales_data/master/2_cleaned_NYC_property_sales.csv')
df = pd.get_dummies(df, columns=['NEIGHBORHOOD', 'BUILDING CLASS AT TIME OF SALE', 'TAX CLASS AT TIME OF SALE'])
df['current_year'] = datetime.datetime.now().year
df['YEAR BUILT'].astype(int)
df['AGE OF BUILDING'] = df['current_year']-df['YEAR BUILT']
bins = [0,3,10,20,30,50,75,100,150,1000]
labels = [1,2,3,4,5,6,7,8,9]
df['AGE OF BUILDING'] = pd.cut(df['AGE OF BUILDING'], bins=bins, labels=labels, right=True)
df['AGE OF BUILDING']=df['AGE OF BUILDING'].astype('object')
df = pd.get_dummies(df, columns=['AGE OF BUILDING'])
df = df.drop(columns=['#','BOROUGH','BLOCK', 'YEAR BUILT', 'current_year', 'TOTAL UNITS', 'LAND SQUARE FEET', 'BUILDING CLASS CATEGORY'])
df_SALEPRICE = df.pop('SALE PRICE')
df['SALE PRICE']=df_SALEPRICE
df=df.rename(columns=lambda x: x.strip())
df.columns = [col.replace('NEIGHBORHOOD_', '') for col in df.columns]
df.columns = [col.replace('BUILDING CLASS AT TIME OF SALE_', '') for col in df.columns]
df.columns = [col.replace('TAX CLASS AT TIME OF SALE_', '') for col in df.columns]
df['SALE PRICE'] = df['SALE PRICE'][np.abs(df['SALE PRICE']-df['SALE PRICE'].mean()) <= (.5*df['SALE PRICE'].std())]
df['SALE PRICE'].replace('', np.nan, inplace=True)
df.dropna(subset=['SALE PRICE'], inplace=True)
df.to_csv(r'C:\Users\nCalo\Documents\Automifai\Research\Coding_Lessons\Git\NYC_apartment_price_project\Data\finished_data.csv', index=False)
