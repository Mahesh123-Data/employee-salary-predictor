import streamlit as st
import numpy as np
import joblib

model = joblib.load('model.pkl')
st.title("Income >50K Prediction")

# Yeh features X_encoded.columns wali order me match karna hai
# Yaha sirf basic input dikhaya, baki bhi isi tarah banao
age = st.number_input('age', 0, 100, 30)
fnlwgt = st.number_input('fnlwgt', 0, 1000000, 50000)
educational_num = st.number_input('educational-num', 1, 16, 10)
# Baki columns ke liye bhi st.number_input ya st.checkbox etc. lagao

# Sab features ek list me daal kar model input banao (order must match X_encoded.columns)
input_data = [age, fnlwgt, educational_num]  # list poora karo

if st.button("Predict"):
    arr = np.array(input_data).reshape(1, -1)
    pred = model.predict(arr)
    if pred[0]:
        st.success("Prediction: Income > 50K")
    else:
        st.info("Prediction: Income ≤ 50K")
