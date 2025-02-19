import streamlit as st
import pandas as pd
import pickle

# Load the preprocessor and trained model
try:
    with open('preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
except FileNotFoundError:
    st.error("Preprocessor pickle file not found. Please ensure 'preprocessor.pkl' is in the same directory as the app.")
    st.stop()

try:
    with open('models/undersampling/Logistic Regression_pipeline.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model pickle file not found. Please ensure 'model.pkl' is in the same directory as the app.")
    st.stop()

import streamlit as st
import pandas as pd

def predict(data):
    # Display the original input data
    st.write("Input data:")
    st.dataframe(data)

    # # Apply the preprocessing pipeline
    # preprocessed_data = preprocessor.transform(data)

    # # Get feature names for numerical features directly
    # numerical_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

    # # Get feature names from OneHotEncoder for categorical features
    # categorical_features = ['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
    # categorical_feature_names = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))

    # # Combine all feature names
    # feature_names = numerical_features + categorical_feature_names

    # # Create a DataFrame with the processed data and new feature names
    # preprocessed_data_df = pd.DataFrame(preprocessed_data, columns=feature_names)

    # # Optionally, display the preprocessed data with named columns
    # st.write("Preprocessed data:")
    # st.dataframe(preprocessed_data_df)

    # Make a prediction
    return model.predict_proba(data)[:, 1]  # Assuming the model returns probabilities


# Title of the application
st.title('Heart Disease Prediction Tool')

# Creating a form for user input
with st.form("input_form"):
    st.write("Please enter the following details:")

    # Fields to collect inputs
    age = st.number_input('Age', min_value=18, max_value=100, value=50, help='Enter your age in years. Age is a primary factor in heart disease risk.')
    height = st.number_input('Height (cm)', min_value=100, max_value=250, value=170, help='Enter your height in centimeters.')
    weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70, help='Enter your weight in kilograms to calculate BMI.')
    male = st.radio('Gender', options=[1, 0], format_func=lambda x: 'Male' if x == 1 else 'Female', help='Select your gender. Gender can influence risk factors for heart disease.')
    education = st.selectbox('Education Level', options=[1, 2, 3, 4], format_func=lambda x: {1: 'Some High School', 2: 'High School Graduate', 3: 'Some College', 4: 'College Graduate'}[x], help='Select the highest level of education you have completed.')
    current_smoker = st.radio('Current Smoker', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No', help='Indicate if you are currently a smoker. Smoking is a significant risk factor for heart disease.')
    cigs_per_day = st.number_input('Cigarettes Per Day', min_value=0, max_value=100, value=10, help='If you smoke, how many cigarettes do you smoke per day?')
    bp_meds = st.radio('On Blood Pressure Medication', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No', help='Indicate if you are taking medication for blood pressure.')
    prevalent_hyp = st.radio('Hypertensive', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No', help='Indicate if you have been diagnosed with hypertension (DE: Bluthochdruck).')
    diabetes = st.radio('Diabetic', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No', help='Indicate if you have diabetes. Diabetes increases the risk of heart disease.')
    prevalent_stroke = st.radio('History of Stroke', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No', help='Indicate if you have had a stroke before. History of stroke can influence your heart disease risk.')

    # Form submission button
    submitted = st.form_submit_button("Submit")
    if submitted:
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        
        # Constructing the data dictionary from the inputs
        sample_data = {
            'age': age,
            'cigsPerDay': cigs_per_day,
            'totChol': 236,  # Default or average value if unknown
            'sysBP': 132,    # Default or average value if unknown
            'diaBP': 82,     # Default or average value if unknown
            'BMI': bmi,
            'heartRate': 75,  # Default or average value if unknown
            'glucose': 81,   # Default or average value if unknown
            'male': male,
            'education': education,
            'currentSmoker': current_smoker,
            'BPMeds': bp_meds,
            'prevalentStroke': prevalent_stroke,
            'prevalentHyp': prevalent_hyp,
            'diabetes': diabetes
        }



        # Convert the data dictionary to DataFrame
        input_df = pd.DataFrame([sample_data])


        # Calling the predict function
        prediction_probability = predict(input_df)

        # Display the prediction result
        st.write('Prediction Probability of Heart Disease:', prediction_probability)
