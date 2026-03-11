import streamlit as st
import pandas as pd
import joblib

# Load models
salary_model = joblib.load("salary_model.pkl")
attrition_model = joblib.load("attrition_model.pkl")

st.title("HR Analytics Prediction System")

age = st.number_input("Age", 18, 60)
joblevel = st.slider("Job Level", 1, 5)
experience = st.number_input("Total Working Years")
years_company = st.number_input("Years At Company")
overtime = st.selectbox("OverTime", ["Yes", "No"])

overtime = 1 if overtime == "Yes" else 0

# Create input dataframe
test_employee = pd.DataFrame(columns=attrition_model.feature_names_in_)
test_employee.loc[0] = 0

test_employee["Age"] = age
test_employee["JobLevel"] = joblevel
test_employee["TotalWorkingYears"] = experience
test_employee["YearsAtCompany"] = years_company
test_employee["OverTime"] = overtime

if st.button("Predict"):

    salary_input = test_employee.drop(columns=["MonthlyIncome"], errors="ignore")

    salary_prediction = salary_model.predict(salary_input)

    st.write("Predicted Salary:", salary_prediction[0])

    test_employee["MonthlyIncome"] = salary_prediction[0]

    attrition_prediction = attrition_model.predict(test_employee)

    if attrition_prediction[0] == 1:
        st.error("Employee likely to leave")
    else:
        st.success("Employee likely to stay")