import streamlit as st
import numpy as np
import joblib

model = joblib.load('model.pkl')
st.title("Income >50K Prediction")

# Features as per your Colab X_encoded.columns list
# Note: Yeh list feature names ka example hai, aap apne full feature list yahan paste kar sakte hain
features = [
    'age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week',
    'workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private',
    'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov',
    'workclass_Without-pay', 'education_11th', 'education_12th', 'education_1st-4th',
    'education_5th-6th', 'education_7th-8th', 'education_9th', 'education_Assoc-acdm',
    'education_Assoc-voc', 'education_Bachelors', 'education_Doctorate', 'education_HS-grad',
    'education_Masters', 'education_Preschool', 'education_Prof-school', 'education_Some-college',
    'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse',
    'marital-status_Married-spouse-absent', 'marital-status_Never-married',
    'marital-status_Separated', 'marital-status_Widowed', 'occupation_Adm-clerical',
    'occupation_Armed-Forces', 'occupation_Craft-repair', 'occupation_Exec-managerial',
    'occupation_Farming-fishing', 'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
    'occupation_Other-service', 'occupation_Priv-house-serv', 'occupation_Prof-specialty',
    'occupation_Protective-serv', 'occupation_Sales', 'occupation_Tech-support',
    'occupation_Transport-moving', 'relationship_Not-in-family', 'relationship_Other-relative',
    'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife',
    'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White', 'gender_Male',
    'native-country_Cambodia', 'native-country_Canada', 'native-country_China',
    'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic',
    'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England',
    'native-country_France', 'native-country_Germany', 'native-country_Greece',
    'native-country_Guatemala', 'native-country_Haiti', 'native-country_Holand-Netherlands',
    'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary',
    'native-country_India', 'native-country_Iran', 'native-country_Ireland',
    'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan',
    'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua',
    'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru',
    'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal',
    'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South',
    'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago',
    'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia'
]

input_values = []

# Define which features are checkbox (dummy variables) vs numeric inputs
# Generally, features starting with these prefixes are dummy variables (0 or 1):
dummy_prefixes = ['workclass_', 'education_', 'marital-status_', 'occupation_', 'relationship_', 'race_', 'gender_', 'native-country_']

for feature in features:
    if any(feature.startswith(prefix) for prefix in dummy_prefixes):
        val = st.checkbox(feature)
        input_values.append(int(val))
    else:
        # For numeric columns, provide reasonable min/max/default values.
        # You can customize min/max/default based on your dataset.
        if feature == 'age':
            val = st.number_input(feature, min_value=10, max_value=100, value=30)
        elif feature == 'fnlwgt':
            val = st.number_input(feature, min_value=0, max_value=1000000, value=50000)
        elif feature == 'educational-num':
            val = st.number_input(feature, min_value=1, max_value=16, value=10)
        elif feature == 'capital-gain':
            val = st.number_input(feature, min_value=0, max_value=100000, value=0)
        elif feature == 'capital-loss':
            val = st.number_input(feature, min_value=0, max_value=100000, value=0)
        elif feature == 'hours-per-week':
            val = st.number_input(feature, min_value=1, max_value=100, value=40)
        else:
            # For unexpected numerical features, give a generic input
            val = st.number_input(feature, value=0)
        input_values.append(val)

if st.button("Predict"):
    input_array = np.array(input_values).reshape(1, -1)
    prediction = model.predict(input_array)
    if prediction[0] == 1:
        st.success("Prediction: Income > 50K")
    else:
        st.info("Prediction: Income ≤ 50K")
