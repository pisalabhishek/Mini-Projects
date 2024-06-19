import streamlit as st
import pandas as pd
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_price(car_company_name, engine_type, car_body, fuel_type, highway_mpg, city_mpg, peak_rpm, horsepower, curb_weight, car_height, car_width, car_length, wheelbase, symboling):
    predicted_price = model.predict([[horsepower, curb_weight, highway_mpg]])[0]
    return predicted_price, car_company_name

st.title("Car Price Prediction Dashboard")

st.sidebar.header("Enter Car Features")

car_company_name = st.sidebar.text_input("Car Company Name", "Toyota")
engine_type = st.sidebar.selectbox("Engine Type", ["OHV", "OHC", "DOHC", "SOHC"])
car_body = st.sidebar.selectbox("Car Body", ["sedan", "hatchback", "wagon", "hardtop", "convertible"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["gas", "diesel"])
highway_mpg = st.sidebar.number_input("Highway MPG", min_value=0)
city_mpg = st.sidebar.number_input("City MPG", min_value=0)
peak_rpm = st.sidebar.number_input("Peak RPM", min_value=0)
horsepower = st.sidebar.number_input("Horsepower", min_value=0)
curb_weight = st.sidebar.number_input("Curb Weight", min_value=0)
car_height = st.sidebar.number_input("Car Height", min_value=0.0)
car_width = st.sidebar.number_input("Car Width", min_value=0.0)
car_length = st.sidebar.number_input("Car Length", min_value=0.0)
wheelbase = st.sidebar.number_input("Wheelbase", min_value=0.0)
symboling = st.sidebar.number_input("Symboling", min_value=0)

if st.sidebar.button("Predict"):
    predicted_price, predicted_car_company_name = predict_price(car_company_name, engine_type, car_body, fuel_type, highway_mpg, city_mpg, peak_rpm, horsepower, curb_weight, car_height, car_width, car_length, wheelbase, symboling)
    st.sidebar.success(f"Predicted Price for {predicted_car_company_name}: ${predicted_price:.2f}")

st.subheader("Dataset")
st.write(pd.read_csv('CarPrice_Assignment.csv'))
