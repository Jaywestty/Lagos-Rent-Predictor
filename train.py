import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

#load data
df = pd.read_csv('lagos-rent 3.csv')

#drop duplicates
df.drop_duplicates(inplace=True)

#Clean the 'Price' column by removing non-numeric characters and also converting the data type to numerical
df['Price'] = pd.to_numeric(
    df['Price'].str.replace(r'[^\d.]', '', regex=True),
    errors='coerce'
)

# Extract numeric values from 'Bedrooms' column also converting the data type to numerical
df['Bedrooms'] = pd.to_numeric(df['Bedrooms'].str.extract(r'(\d+)')[0], errors='coerce')

# Extract numeric values from 'Toilets' column also converting the data type to numerical

df['Toilets'] = pd.to_numeric(df['Toilets'].str.extract(r'(\d+)')[0], errors='coerce')

# Extract numeric values from 'Bathroom' column also converting the data type to numerical

df['Bathrooms'] = pd.to_numeric(df['Bathrooms'].str.extract(r'(\d+)')[0], errors='coerce')

#handle outliers with IQR-based Capping
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap outliers instead of removing them
df['Price'] = df['Price'].clip(lower=lower_bound, upper=upper_bound)


#create an area score column
def get_area_score(neighborhood):
    if pd.isna(neighborhood):
        return 2  # Neutral score for missing values

    area = str(neighborhood).lower()

    # Define score tiers in a dictionary
    score_map = {
        5: [
            'banana island', 'bourdillon', 'victoria island extension'
        ],
        4: [
            'lekki phase 1', 'old ikoyi', 'osborne foreshore estate', 'parkview estate',
            'dolphin estate', 'victoria island', '1004', 'ikoyi', 'oniru',
            'adeola odeku', 'ligali ayorinde', 'awolowo road'
        ],
        3: [
            'chevron', 'lekki scheme', 'other lekki', 'agungi', 'osapa london', 'ologolo',
            'ajah', 'ikate', 'ikota', 'ado', 'sangotedo', 'badore',
            'allen avenue', 'opebi', 'oregun'
        ],
        2: [
            'yaba', 'other yaba', 'gbagada', 'other gbagada', 'ikeja', 'other ikeja',
            'surulere', 'other surulere', 'toyin street', 'adeniyi jones',
            'alausa', 'awolowo way', 'omole phase 1', 'omole phase 2',
            'magodo gra phase 1', 'magodo', 'ojodu berger', 'other ojodu',
            'isheri north', 'bode thomas', 'randle avenue', 'onike'
        ],
        1: [
            'oworonshoki', 'lawanson', 'ijesha', 'adelabu', 'aguda', 'ogunlana',
            'marsha', 'ojuelegba', 'ebute-metta', 'sabo', 'alagomeji', 'jibowu',
            'akoka', 'ifako', 'soluyi', 'medina', 'millennium ups'
        ]
    }

    # Match neighborhood to score
    for score, locations in score_map.items():
        for name in locations:
            if name in area:
                return score

    return 2  # Default to mid-score for unknown/unclassified areas
  
  
df['Area_Score'] = df['Neighborhood'].apply(get_area_score)

#Count Luxury features and create a new column
df['Luxury_Count'] = df['Serviced'] + df['Furnished'] + df['Newly Built']

#Combine area + luxury into one score
df['Location_Premium'] = df['Area_Score'] + (df['Luxury_Count'] * 0.5)

#select needed columns
num_coll = [ 'Serviced', 'Newly Built', 'Furnished', 'Bedrooms',
            'Bathrooms','Area_Score','Luxury_Count','Location_Premium']
cat_coll = ['City']

#Split into feature and target variables
X = df.drop(columns=['Price'])
y = df['Price']

#Split into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create a preprocessing pipeline


numerical_transformer = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='median')),
    ('Scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_coll),
    ('cat', categorical_transformer, cat_coll)
])

#Build model
model = XGBRegressor(random_state=42,n_estimators=200, learning_rate = 0.1)

#Build pipline
pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('Regressor', model)
    ])
pipeline.fit(X_train, y_train)

#Save the model
with open('Regressor.pkl', 'wb') as f:
    joblib.dump(pipeline, f)
print('Model saved succesfully')

sample = pd.DataFrame([{
    'Serviced': 0,
    'Newly Built': 1,
    'Furnished': 0,
    'Bedrooms': 3,
    'Bathrooms': 1,
    'City': 'lekki',
    'Area_Score': 3,
    'Luxury_Count':1,
    'Location_Premium':3.5
    
}])

predy = pipeline.predict(sample)[0]
print(f'Price of house is {predy}')