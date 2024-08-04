import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv('gld_price_data.csv')

# Prepare data
X = data[['SPX', 'USO', 'SLV', 'EUR/USD']]
y = data['GLD']

# Standardize the data
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
reg = RandomForestRegressor()
reg.fit(X_train, y_train)

# Streamlit App
st.title("Gold Price Prediction App")

# Input features
spx = st.number_input('SPX', min_value=0.0, format="%.4f")
uso = st.number_input('USO', min_value=0.0, format="%.4f")
slv = st.number_input('SLV', min_value=0.0, format="%.4f")
eur_usd = st.number_input('EUR/USD', min_value=0.0, format="%.4f")

# Prediction
if st.button("Predict Gold Price"):
    input_data = np.array([spx, uso, slv, eur_usd]).reshape(1, -1)
    # input_data_scaled = sc.transform(input_data)
    prediction = reg.predict(input_data)
    st.write(f"The predicted price of gold is: {prediction[0]:.2f} USD")
