# Employee Attrition Prediction using Machine Learning 📊

## 📌 Project Overview
Employee Attrition Prediction is a Machine Learning project that predicts whether an employee is likely to leave an organization based on employee-related information.

The project uses employee data to analyze attrition patterns and builds classification models to identify employees who may be at risk of leaving.

## 🎯 Project Objectives
- Analyze employee attrition patterns.
- Perform data cleaning and preprocessing.
- Explore factors associated with employee attrition.
- Train and compare Machine Learning models.
- Predict the likelihood of employee attrition.

## 📂 Dataset
- **Dataset:** IBM HR Analytics Employee Attrition Dataset
- **Total Records:** 1,470
- **Features Used:** 30
- **Target Variable:** `Attrition`
- **Target Classes:** Yes and No

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Joblib

## 🤖 Machine Learning Models
The following models were trained and compared:
- Logistic Regression
- Random Forest
- Balanced Random Forest

**Final Model:** Logistic Regression with a classification threshold of 0.4.

## 📈 Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 73.13% |
| Precision | 34.91% |
| Recall | 78.72% |
| F1 Score | 48.37% |
| ROC-AUC | 80.32% |

The classification threshold was adjusted to 0.4 to improve the model's ability to identify employees who may leave.

## 🔍 Project Workflow
1. Data loading and understanding
2. Exploratory Data Analysis (EDA)
3. Data cleaning and preprocessing
4. Feature encoding and scaling
5. Train-test split
6. Model training
7. Model comparison and evaluation
8. Threshold optimization
9. Feature importance analysis
10. Model saving using Joblib
11. Streamlit application development

## 📁 Project Files
- `Employee_Attrition_Prediction.ipynb` — Jupyter Notebook containing data analysis, preprocessing, model training, and evaluation.
- `app.py` — Streamlit web application for making predictions.
- `employee_attrition_model.pkl` — Saved final model and classification threshold.
- `WA_Fn-UseC_-HR-Employee-Attrition.csv` — Dataset used for the project.

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone <your-github-repository-url>
cd <your-repository-folder>
```

### 2. Install Required Libraries
```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### 3. Run the Streamlit Application
Make sure `app.py` and `employee_attrition_model.pkl` are in the same folder.

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal to use the application.

## 💡 Key Learning Outcomes
- Data preprocessing and exploratory analysis
- Handling an imbalanced classification dataset
- Building and comparing classification models
- Evaluating models using precision, recall, F1 score, and ROC-AUC
- Saving a trained model and using it in a Streamlit app

