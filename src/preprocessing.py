from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from src import config

# Numeric Features
NUMERIC_FEATURES = [
    "Age",
    "Year_of_study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
]

# Nominal features
NOMINAL_FEATURES = [
    "Gender"
]

# Ordinal features
ORDINAL_FEATURES = [
    "Physical_Activity",
    "Diet_Quality"
]

# Binary features
BINARY_FEATURES = [
    "Part_Time_Job",
    "Previous_Mental_Health_History"
]

# Feature
MODEL_FEATURES = (
    NUMERIC_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES
)

def build_preprocessor() -> ColumnTransformer:

    # numeric_pipeline
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
        ]
    )

    # nominal_pipeline
    nominal_pipeline = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False)),
        ]
    )

    # Ordinal_Pipeline
    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("encoder",OrdinalEncoder(categories=[
                ["Low","Moderate","High"],
                ["Poor","Average","Good"],
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            ),
            ),
        ]
    )

    # Binary_Pipeline
    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[
                        ["No", "Yes"],
                        ["No", "Yes"],
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric",numeric_pipeline,NUMERIC_FEATURES),
            ("nominal",nominal_pipeline,NOMINAL_FEATURES),
            ("ordinal",ordinal_pipeline,ORDINAL_FEATURES),
            ("binary",binary_pipeline,BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor

