import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression

#data

df = pd.read_csv("dastgah.csv")

#regression

x = df.drop("failure", axis=1)
y =  df["failure"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
standard_reg =  StandardScaler()
x_train_scaled = standard_reg.fit_transform(x_train)

reg_model = LinearRegression()
reg_model.fit(x_train_scaled, y_train)

#calssification

x1 = df.drop("status", axis=1)
y1 = df["status"]

x_train1, x_test1, y_train1, y_test1 = train_test_split(
    x1,
    y1,
    test_size=0.3,
    random_state=42
)
standard_clf = StandardScaler()
x_train1_scaled = standard_clf.fit_transform(x_train1)

clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(x_train1_scaled, y_train1)

#UI
st.title("Industrial Machine Prediction")

st.write("Enter Machine Data")

temp = st.slider("temperature", 40, 120, 75)
vibration = st.slider("vibration", 0.0, 10.0, 4.0)
pressure = st.slider("pressure", 100, 300, 180)
oil = st.slider("oil", 0, 100, 60)
hours = st.slider("hours", 1000, 2000, 1200)
error_count = st.slider("error count", 0, 20, 5)

if st.button("predict"):
    
    #regression input
    reg_input = pd.DataFrame({
        "temp" : [temp],
        "vibration" : [vibration],
        "pressure" : [pressure],
        "oil" : [oil],
        "hours" : [hours],
        "error_count" : [error_count],
        "status" : [0]
        
    })
    reg_input_scaled = standard_reg.transform(reg_input)
    days_prediction = reg_model.predict(reg_input_scaled)

    #classification input
    clf_input = pd.DataFrame({
        "temp" : [temp],
        "vibration" : [vibration],
        "pressure" : [pressure],
        "oil" : [oil],
        "hours" : [hours],
        "error_count" : [error_count],
        "failure" : [days_prediction[0]]
        
    })
    clf_input_scaled = standard_clf.transform(clf_input)
    status_prediction = clf_model.predict(clf_input_scaled)

    #output
    st.subheader("prediction result")

    if status_prediction[0] == 1 :
        st.error("machine will fail soon")
    else:
        st.success("machine is healthy")
    st.info(f"estimated days to failure{days_prediction[0]:.2f} days")  