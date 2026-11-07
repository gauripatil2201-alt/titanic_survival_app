import joblib
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Titanic survival Prediction")
st.title("🚢Titanic Survival Prediction")

@st.cache_resource
def load_model():
    model=joblib.load("model.joblib")
    features=joblib.load("features.joblib")
    return model,features

try:
    if not os.path.exists("model.joblib"):
        st.error("Model file doesnot exists.")
        st.stop()
    model,features=load_model()
except Exception as e:
    st.error(f"Unable to load model:{e}")
    st.stop()

st.subheader("PASSENGER INFORMATION")

sex=st.radio("Sex",["Female","Male"])
age=st.slider("Age",min_value=0,max_value=80,value=30)
fare=st.slider("Fare",min_value=0.0,max_value=600.0,value=50.0)
pclass=st.selectbox("Passenger Class",[1,2,3])

sex_encoded=1 if sex=="Male" else 0

if st.button("Predict Survival"):
    try:
        input_data=pd.DataFrame(
            [[sex_encoded,age,fare,pclass]],
            columns=features
        )

        prediction=model.predict(input_data)[0]
        probability=model.predict_proba(input_data)[0]

        survival_probability=probability[1]*100

        if prediction==1:
            st.success("Passenger is likely to survive.")
        else:
            st.warning("Passenger is unlikely to survive.")

        st.metric("Survival Probability",f"{survival_probability:.2f}%")
    except Exception:
        st.error("Invalid input.Please check the values and try again.")
st.caption(
    """This model uses Decision Tree Classifier trained on Titanic dataset.
    Predictions are based on Sex,Age,Fare, and Passenger Class.
    This application is for educational purpose only and should not be used for real life decision making
    """
)
        