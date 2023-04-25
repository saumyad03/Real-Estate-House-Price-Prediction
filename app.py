import streamlit as st
import lightgbm as ltb
import pandas as pd
import shap
import matplotlib
from streamlit_shap import st_shap

features = [
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
    "YearRemodAdd",
    "BsmtFinSF1",
    "Fireplaces",
    "1stFlrSF",
    "GarageYrBlt",
    "2ndFlrSF"
    ]
    
st.title("Real Estate House Price Prediction Model")
slider1 = st.slider("Above Ground Living Area Square Feet", 334, 5642) #GrLivArea
slider2 = st.slider("Size of Garage (Car Capacity)", 0, 4) #GarageCars
slider3 = st.slider("Total Basement Area Square Feet", 0, 6110) #TotalBsmtSF
slider4 = st.slider("Original Construction Date", 1872, 2010) #YearBuilt
slider5 = st.slider("Remodel Date", 1950, 2010) #YearRemodAdd
slider6 = st.slider("Basement Type 1 Finished Square Feet", 0, 5644) #BsmtFinSF1
slider7 = st.slider("Number of Fireplaces", 0, 3) #Fireplaces
slider8 = st.slider("First Floor Square Feet", 344, 4692) #1stFlrSF
slider9 = st.slider("Year Garage Was Built", 1900, 2010) #GargageYrBlt
slider10 = st.slider("Second Floor Square Feet", 0, 2065) #2ndFlrSF

sliders = [slider1, slider2, slider3, slider4, slider5, slider6, slider7, slider8, slider9, slider10]

def calculate1():
    model = ltb.Booster(model_file='model.txt')
    data = {}
    for i in range(len(features)):
        data[features[i]] = [sliders[i]]
    df = pd.DataFrame(data)
    st.write("House Price: ${:.2f}".format(model.predict(df)[0]))
    model.params["objective"] = "regression"
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df)
    st_shap(shap.summary_plot(shap_values, df))
    shap_interaction = shap.TreeExplainer(model).shap_interaction_values(df)
    st_shap(shap.summary_plot(shap_interaction, df))
    
def calculate2():
    model = ltb.Booster(model_file='oldmodel.txt')
    data = {}
    for i in range(len(features)):
        data[features[i]] = [sliders[i]]
    df = pd.DataFrame(data)
    st.write("House Price: ${:.2f}".format(model.predict(df)[0]))
    model.params["objective"] = "regression"
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df)
    st_shap(shap.summary_plot(shap_values, df))
    shap_interaction = shap.TreeExplainer(model).shap_interaction_values(df)
    st_shap(shap.summary_plot(shap_interaction, df))

    
calculator1 = st.button("Calculate House Price (tuned model)", on_click=calculate1)
calculator2 = st.button("Calculate House Price (untuned model)", on_click=calculate2)