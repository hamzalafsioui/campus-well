import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CampusWell - Student Mental Health Risk",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_pipeline.joblib"
EVAL_PATH = ROOT / "models" / "evaluation.json"       # written by Airflow
METRICS_PATH = ROOT / "reports" / "final_metrics.json"   # written by notebooks


# --------------------------------------- Helpers -------------------------------
def load_metrics():
    """Return the most recent evaluation JSON (Airflow > notebook fallback)."""
    for p in (EVAL_PATH, METRICS_PATH):
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f), p.name
    return None, None


# --------------------------------------- Sidebar ----------------------------
st.sidebar.title("CampusWell")
st.sidebar.caption("Educational prototype - not a diagnostic tool")

page = st.sidebar.radio("Navigation", ["Prediction", "Model Performance", "About"])

# ============================
# PAGE  PREDICTION
# ============================
if page == "Prediction":
    st.title("Student Mental Health Risk Assessment")
    st.info("Enter the student details below, then click Predict.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("General Profile")
        age = st.slider("Age", 18, 30, 21)
        year_of_study = st.selectbox("Year of Study", [1, 2, 3, 4, 5], index=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
        part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])
        previous = st.selectbox("Mental Health History", ["No", "Yes"])

    with col2:
        st.subheader("Academic")
        gpa = st.slider("GPA", 1.5, 4.0, 3.2, step=0.05)
        study_hours = st.slider("Study Hours / Week", 5, 55, 25)
        attendance = st.slider("Attendance Rate (%)", 30, 100, 85)
        physical = st.selectbox("Physical Activity", ["Low", "Moderate", "High"], index=1)
        diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"], index=1)

    with col3:
        st.subheader("Well-being")
        sleep = st.slider("Sleep Hours / Night", 3.5, 9.5, 7.0, step=0.5)
        screen_time = st.slider("Screen Time (hours/day)", 1.0, 14.0, 6.0, step=0.5)
        social_media = st.slider("Social Media (hours/day)", 0.5, 10.0, 3.0, step=0.5)
        financial = st.slider("Financial Stress (1-10)", 1.0, 10.0, 5.0, step=0.5)
        family = st.slider("Family Support (1-10)", 1.0, 10.0, 7.0, step=0.5)

    input_df = pd.DataFrame([{
        "Age": age, "Year_of_Study": year_of_study,
        "Sleep_Hours": sleep, "Study_Hours_Per_Week": study_hours,
        "GPA": gpa, "Social_Media_Hours": social_media,
        "Screen_Time_Hours": screen_time, "Financial_Stress": financial,
        "Family_Support": family, "Attendance_Rate": attendance,
        "Gender": gender, "Physical_Activity": physical,
        "Diet_Quality": diet, "Part_Time_Job": part_time_job,
        "Previous_Mental_Health_History": previous,
    }])

    if st.button("Predict Risk", type="primary", use_container_width=True):
        if not MODEL_PATH.exists():
            st.warning("Model not found. Please run the Airflow DAG 'campuswell_pipeline' to train the model.")
        else:
            pipeline = joblib.load(MODEL_PATH)
            prediction = pipeline.predict(input_df)[0]
            proba = pipeline.predict_proba(input_df)[0]
            classes = list(pipeline.classes_)

            st.divider()
            st.subheader(f"Predicted Risk Level: {prediction}")

            prob_df = pd.DataFrame({
                "Risk Level": classes,
                "Probability (%)": [round(p * 100, 1) for p in proba],
            }).set_index("Risk Level")
            st.bar_chart(prob_df)

            st.caption("Disclaimer: Strictly for educational and research purposes. Do not use for clinical diagnosis or screening.")

# ===========================
# PAGE  MODEL PERFORMANCE
# ===========================
elif page == "Model Performance":
    st.title("Model Performance")

    metrics, source = load_metrics()

    if metrics is None:
        st.info("No metrics file found. Run the Airflow DAG to generate evaluation results.")
        st.stop()

    st.caption(f"Source: `{source}`")

    # -- Summary cards
    tm = metrics.get("test_metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{tm.get('accuracy', 0)*100:.1f}%")
    c2.metric("F1 Macro", f"{tm.get('f1_macro', 0):.3f}")
    c3.metric("ROC-AUC", f"{tm.get('roc_auc_ovr', 0):.3f}")
    c4.metric("Test Samples", metrics.get("n_test", "N/A"))

    st.divider()

    # -- Confusion matrix
    cm = metrics.get("test_confusion_matrix") or metrics.get("confusion_matrix")
    classes = metrics.get("classes", ["Low", "Medium", "High"])

    if cm:
        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        cm_df.index.name = "Actual \\ Predicted"
        st.dataframe(cm_df, use_container_width=True)

    # -- Classification report
    report = metrics.get("test_classification_report") or metrics.get("classification_report")
    if report:
        st.subheader("Classification Report")
        rows = {k: v for k, v in report.items() if isinstance(v, dict)}
        st.dataframe(pd.DataFrame(rows).T.round(3), use_container_width=True)

    # -- Best params
    st.subheader("Model and Hyperparameters")
    st.write(f"**Model:** {metrics.get('model_name', 'N/A')}")
    st.json(metrics.get("best_params", {}))
