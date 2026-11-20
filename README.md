# CampusWell: Student Mental Health Risk Analysis and Prediction

CampusWell is an educational, end-to-end machine learning platform for analyzing student well-being factors and estimating mental health risk levels. It combines data exploration, leakage-safe preprocessing, model comparison, hyperparameter tuning, automated Airflow workflows, and an interactive Streamlit dashboard to demonstrate how responsible predictive analytics can support early academic and counseling support. The project is intended for learning and research—not clinical diagnosis or decision-making.

## Important Disclaimer

This project is strictly educational. The predictive models developed here are educational prototypes and must never be used for clinical diagnosis, psychiatric screening, disciplinary decisions, or student admissions.

## Project Objectives

- Risk Detection: Identify students with elevated risk of psychological distress for early academic and counseling support.
- Prevention: Anticipate student well-being needs and provide timely support mechanisms.
- Factor Analysis: Study correlations between academic habits, lifestyle metrics, and reported stress levels.
- Pipeline Automation: Automate data preprocessing, model training, hyperparameter tuning, and model evaluation using Apache Airflow.
- Interactive Interface: Provide an interactive web application built with Streamlit to review model metrics and run sample risk evaluations.

## Machine Learning Lifecycle

1. Data Exploration and Auditing:
   Analyze distributions, identify missing data and duplicate records, compute summary statistics, and evaluate class balance for the target risk variable (Low, Medium, High).

2. Data Preparation and Leakage Prevention:
   Split data into an 80 percent training set and a 20 percent test set. Categorical variables are encoded and numerical features are standardized within a scikit-learn pipeline to prevent data leakage.

3. Model Training and Comparison:
   Train and compare multiple classifiers using stratified 5-fold cross-validation on identical training folds:
   - Dummy Classifier (baseline)
   - Logistic Regression
   - Random Forest Classifier
   - Gradient Boosting Classifier

4. Optimization and Evaluation:
   Perform hyperparameter tuning via GridSearchCV on candidate models, select the top-performing configuration based on macro F1 score, evaluate performance on the held-out test set, and serialize the final pipeline to disk.

5. Orchestration and Visualization:
   Orchestrate all pipeline stages using an Apache Airflow DAG and expose results and predictions through a Streamlit dashboard.

## Project Structure

- airflow_home/
  - dags/campuswell_pipeline_dag.py: Automated pipeline DAG.
  - airflow.cfg: Airflow configuration file.
  - webserver_config.py: Web UI security configuration.
- app/
  - streamlit_app.py: Multi-page interactive Streamlit dashboard.
- data/
  - raw/student_mental_health.csv: Source dataset.
  - interim/: Cleaned train and test data splits.
- models/
  - final_pipeline.joblib: Serialized production pipeline.
  - evaluation.json: Test set performance metrics and confusion matrix.
  - cv_comparison.json: Cross-validation comparison summary.
- notebooks/
  - 01_eda.ipynb: Exploratory data analysis notebook.
  - 02_preprocessing.ipynb: Data cleaning and transformation tests.
  - 03_modeling.ipynb: Model experimentation and evaluation.
- src/
  - config.py: Project paths, constants, and column definitions.
  - data.py: Ingestion, cleaning, and train-test splitting logic.
  - preprocessing.py: Scikit-learn transformer pipelines.
  - models.py: Model registries, hyperparameter grids, and CV definitions.
  - evaluation.py: Metric calculation and report generation utilities.
- docker-compose.yml: Multi-service container orchestration.
- Dockerfile: Base environment definition with Python 3.12.
- requirements.txt: Python project dependencies.

## Services and Ports

The environment runs three core containerized services:

- Streamlit Web App:
  URL: http://localhost:8502
  Purpose: Interactive prediction and model evaluation interface.

- Apache Airflow:
  URL: http://localhost:8081
  Purpose: Scheduled and on-demand ML pipeline workflow orchestration.
  Username: admin

- JupyterLab:
  URL: http://localhost:8888
  Purpose: Interactive notebook development and analysis.

## Getting Started with Docker

1. Start all services in the background:
   docker compose up -d

2. View container status:
   docker compose ps

3. Follow service logs:
   docker compose logs -f

4. Stop all running services:
   docker compose down

## Running the Airflow Pipeline

1. Open the Airflow Web UI at http://localhost:8081.
2. Sign in with the username admin.
3. Locate the DAG named campuswell_pipeline.
4. Toggle the DAG switch to active, then click the trigger button to execute the full pipeline.
5. The pipeline automatically runs the following tasks in sequence:
   prepare_data -> train_models -> optimize_model -> evaluate_model -> save_pipeline

## Exploring the Streamlit Application

Open http://localhost:8502 in your browser to access:

- Prediction: Input student demographic, lifestyle, and academic features to estimate risk level and view class probability distributions.
- Model Performance: Review test set accuracy, macro F1, ROC-AUC, confusion matrix, and hyperparameter configurations.
- About: View project scope, methodology, and the pedagogical disclaimer.

