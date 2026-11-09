from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

for folder in (DATA_RAW, DATA_INTERIM, FIGURES, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

# Filename of the Kaggle CSV
RAW_FILENAME = "student_mental_health.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

ID_COL = "Student_ID"
TARGET_COL = "Mental_Health_Risk"
TARGET_ORDER = ["Low", "Medium", "High"]

# (min, max) for each numeric column
NUMERIC_RANGES = {
    "Age": (18, 30),
    "Year_of_Study": (1, 5),
    "Sleep_Hours": (3.5, 9.5),
    "Study_Hours_Per_Week": (5, 55),
    "GPA": (1.5, 4.0),
    "Social_Media_Hours": (0.5, 10.0),
    "Screen_Time_Hours": (1.0, 14.0),
    "Financial_Stress": (1.0, 10.0),
    "Family_Support": (1.0, 10.0),
    "Attendance_Rate": (30, 100),
}

# Allowed categories per column 
CATEGORICAL_VALUES = {
    "Gender": ["Male", "Female", "Non-binary"],
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Good"],
    "Part_Time_Job": ["Yes", "No"],
    "Previous_Mental_Health_History": ["Yes", "No"],
}
