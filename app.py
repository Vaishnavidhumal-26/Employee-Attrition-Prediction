import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# Employee Attrition Prediction - Streamlit App
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("Employee Attrition Prediction")
st.write(
    "Enter the employee details below to estimate the likelihood "
    "of employee attrition using the trained Logistic Regression model."
)

MODEL_PATH = Path(__file__).with_name("employee_attrition_model.pkl")

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Keep employee_attrition_model.pkl "
        "in the same folder as app.py."
    )
    st.stop()

try:
    saved_data = joblib.load(MODEL_PATH)
    model = saved_data["model"]
    threshold = float(saved_data["threshold"])
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

st.caption(f"Prediction threshold: {threshold:.2f}")

with st.form("employee_details"):
    st.subheader("Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        business_travel = st.selectbox(
            "Business Travel",
            ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
        )
        daily_rate = st.number_input("Daily Rate", min_value=0, value=800)
        department = st.selectbox(
            "Department",
            ["Sales", "Research & Development", "Human Resources"]
        )
        distance_from_home = st.number_input(
            "Distance From Home", min_value=1, max_value=50, value=10
        )
        education = st.selectbox("Education (1–5)", [1, 2, 3, 4, 5], index=2)
        education_field = st.selectbox(
            "Education Field",
            [
                "Life Sciences", "Medical", "Marketing",
                "Technical Degree", "Human Resources", "Other"
            ]
        )
        environment_satisfaction = st.selectbox(
            "Environment Satisfaction (1–4)", [1, 2, 3, 4], index=2
        )
        gender = st.selectbox("Gender", ["Female", "Male"])
        hourly_rate = st.number_input(
            "Hourly Rate", min_value=0, max_value=100, value=60
        )
        job_involvement = st.selectbox(
            "Job Involvement (1–4)", [1, 2, 3, 4], index=2
        )
        job_level = st.selectbox("Job Level (1–5)", [1, 2, 3, 4, 5], index=1)
        job_role = st.selectbox(
            "Job Role",
            [
                "Sales Executive", "Research Scientist",
                "Laboratory Technician", "Manufacturing Director",
                "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director",
                "Human Resources"
            ]
        )
        job_satisfaction = st.selectbox(
            "Job Satisfaction (1–4)", [1, 2, 3, 4], index=2
        )
        marital_status = st.selectbox(
            "Marital Status", ["Single", "Married", "Divorced"]
        )

    with col2:
        monthly_income = st.number_input(
            "Monthly Income", min_value=0, value=5000
        )
        monthly_rate = st.number_input(
            "Monthly Rate", min_value=0, value=15000
        )
        num_companies_worked = st.number_input(
            "Number of Companies Worked", min_value=0, value=2
        )
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        percent_salary_hike = st.number_input(
            "Percent Salary Hike", min_value=0, max_value=100, value=15
        )
        performance_rating = st.selectbox(
            "Performance Rating (1–4)", [1, 2, 3, 4], index=2
        )
        relationship_satisfaction = st.selectbox(
            "Relationship Satisfaction (1–4)", [1, 2, 3, 4], index=2
        )
        stock_option_level = st.selectbox(
            "Stock Option Level (0–3)", [0, 1, 2, 3]
        )
        total_working_years = st.number_input(
            "Total Working Years", min_value=0, max_value=50, value=5
        )
        training_times_last_year = st.number_input(
            "Training Times Last Year", min_value=0, max_value=10, value=3
        )
        work_life_balance = st.selectbox(
            "Work Life Balance (1–4)", [1, 2, 3, 4], index=2
        )
        years_at_company = st.number_input(
            "Years At Company", min_value=0, max_value=50, value=2
        )
        years_in_current_role = st.number_input(
            "Years In Current Role", min_value=0, max_value=50, value=1
        )
        years_since_last_promotion = st.number_input(
            "Years Since Last Promotion", min_value=0, max_value=50, value=1
        )
        years_with_curr_manager = st.number_input(
            "Years With Current Manager", min_value=0, max_value=50, value=1
        )

    submitted = st.form_submit_button(
        "Predict Employee Attrition", type="primary", use_container_width=True
    )

if submitted:
    employee = {
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    }

    employee_df = pd.DataFrame([employee])

    try:
        probability = float(model.predict_proba(employee_df)[0][1])
        st.subheader("Prediction Result")
        st.metric("Estimated Attrition Probability", f"{probability * 100:.2f}%")
        st.progress(min(max(probability, 0.0), 1.0))

        if probability >= threshold:
            st.warning("Prediction: Likely to Leave")
        else:
            st.success("Prediction: Likely to Stay")

        st.caption(
            "This is a model-based estimate, not a certainty. "
            "Use it only as a support for further, fair HR review."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "Check that the saved model was created from the same notebook "
            "and uses the same 30 input features."
        )
