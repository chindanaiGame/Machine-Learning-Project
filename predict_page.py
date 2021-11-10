
from operator import index
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as rfr


df = pd.read_csv('insurance.csv')


df.smoker.replace(('yes', 'no'), (1, 0), inplace=True)
df.sex.replace(('male', 'female'), (1, 0), inplace=True)
df.region = pd.Categorical(df.region)
df['region'] = df.region.cat.codes

x = df.drop(["charges"],axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4)

model = LinearRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)

def show_predict_page():
    st.title("Medical Insurance Cost")

    st.write("""### We need some information to Predict for new customer""")

    name = st.text_input("Name")

    age = st.number_input("Age", min_value=0, max_value=100,step=1)

    sex = st.number_input("Sex(0 = No,1 = Yes)", min_value=0, max_value=1,step=1)

    bmi = st.number_input("BMI", min_value=0)

    children = st.number_input("BMI", min_value=0, max_value=5,step=1)

    smoker = st.number_input("smoker(0 = No,1 = Yes)", min_value=0, max_value=1,step=1) 

    region = st.number_input("region(1 = southwest,2 = southeast,3 = northwest,4 = northeast)", min_value=1, max_value=4,step=1)

    ok = st.button("Analysis")
    if ok:
        X = np.array([[age, sex, bmi, children, smoker, region]])
        X = X.astype(float)

        pred_customer_df = model.predict(X)

        st.subheader("Medical Insurance cost for "+ name +f" is : {pred_customer_df[0]:.2f}")
        

    