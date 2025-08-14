# Lagos House Rent Estimator 

A machine learning-powered web app that predicts estimated rental prices for houses in Lagos, Nigeria, based on features such as location, number of bedrooms, bathrooms, and property type. This project aims to help renters, landlords, and real estate agents make informed decisions using data-driven insights.

## Features
* Interactive Web Interface – User-friendly design for quick predictions.
* Accurate Predictions – Trained on real Lagos rental data.
* Multiple Inputs – Location, bedrooms, bathrooms, and property type.
* Instant Results – Get estimated rent within seconds.

## Tech Stack
* Programming Language: Python
* Frontend Framework: Streamlit
* Machine Learning: Scikit-learn
* Data Processing: Pandas, NumPy
* Visualization: Matplotlib, Seaborn
* Deployment: Streamlit Cloud / Local

## Model Training
* Data was cleaned, processed, and encoded.
* Features like location, bedrooms, bathrooms, and property type were used.
* Multiple regression models were tested — Xgboost gave the best performance.
* Model saved as model.pkl for deployment.

## Deployment

The app is deployed on Streamlit Cloud for easy public access.
You can try it live here: https://lagos-rent-predictor.streamlit.app/
