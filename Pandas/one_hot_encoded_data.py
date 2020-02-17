import pandas as pd
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

clean_data = df
print(clean_data.head(10))
