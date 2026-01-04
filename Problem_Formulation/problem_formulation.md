# **Problem Formulation**

## **Business Problem**  
Cardiovascular diseases (CVDs) are among the leading causes of mortality worldwide, placing a significant burden on healthcare systems, patients, and economies. Early identification of individuals at high risk of heart disease enables timely medical intervention, reduces treatment costs, and improves patient outcomes. Traditional diagnostic approaches often rely on manual assessment and clinical expertise, which may be time-consuming and prone to variability.

With the increasing availability of patient health data, machine learning–based predictive systems can assist clinicians by providing data-driven risk assessments. Building a scalable, automated, and reproducible machine learning pipeline for heart disease prediction is therefore essential to support preventive healthcare and clinical decision-making in real-world production environments.

## **Business Objectives**  
1. Develop a machine learning–based classification system to predict the presence or absence of heart disease using patient health attributes.

2. Automate the end-to-end data processing, feature engineering, model training, and evaluation pipeline using MLOps best practices.

3. Enable experiment tracking, model versioning, and reproducibility to support continuous model improvement.

4. Deploy the trained model as a production-ready, cloud-deployable API with monitoring and logging capabilities.

## **Key Data Sources and Attributes**  
The solution utilises the Heart Disease UCI Dataset obtained from the UCI Machine Learning Repository. The dataset contains structured clinical data describing patient health indicators.

### Clinical and Demographic Features
- Age
- Sex
- Chest pain type (cp)
- Resting blood pressure (trestbps)
- Serum cholesterol (chol)
- Fasting blood sugar (fbs)
- Resting electrocardiographic results (restecg)
- Maximum heart rate achieved (thalach)
- Exercise-induced angina (exang)
- ST depression induced by exercise (oldpeak)
- Slope of peak exercise ST segment (slope)
- Number of major vessels colored by fluoroscopy (ca)
- Thalassemia status (thal)

### Target Variable
- Heart Disease Presence (num): Binary classification; presence or absence of heart disease

### Data Quality and Validation Logs
- Missing value counts
- Data type validation
- Outlier detection
- Duplicate record checks

## **Expected Outputs**  
1. Cleaned and preprocessed datasets ready for exploratory data analysis and model training.
2. Engineered and transformed features stored in a structured format for reproducibility.
3. Trained, evaluated, and versioned machine learning models logged using MLflow.
4. A fully automated ML pipeline integrating data ingestion, validation, training, and evaluation.
5. A containerized, production-ready prediction API exposing a /predict endpoint.
6. CI/CD workflows with automated linting, unit testing, model training, and artifact logging.
7. Deployed and monitored model service with request logging and basic performance monitoring. 

## **Evaluation Metrics**  
To assess model performance, the following metrics will be used:  
  - **Accuracy:** Measures how many predictions were correct overall.
  - **Precision:** Indicates how many of the predicted positives are actually correct.
  - **Recall:** Shows how well the model captures actual positives.
  - **F1 Score:** Balances precision and recall, especially useful for imbalanced data.
  - **ROC-AUC Curve:** Evaluates the model's ability to distinguish between classes across different thresholds.
