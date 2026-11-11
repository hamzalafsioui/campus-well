from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src import config

SCORING = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "roc_auc_ovr"
]

def get_cv():
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.RANDOM_STATE
    )

def get_models():

    return {
        # Baseline - Dummy (random guesses that follow the class proportions)
        "Dummy (baseline)":DummyClassifier(strategy="stratified",random_state=config.RANDOM_STATE),

        # logistic regression (linear & interpretable model)
        "Logistic Regression" : LogisticRegression(max_iter=1000,random_state=config.RANDOM_STATE),

        # Random Forest (many independent decision trees that vote together)
        "Random Forest" : RandomForestClassifier(n_estimators=200,random_state=config.RANDOM_STATE),

        # Gradient Boosting (trees trained one after another each correction the previous ones)
        "Gradient Boosting":GradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }


PARAM_GRIDS = {
    "Logistic Regression": {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__class_weight": [None, "balanced"],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200, 300, 500],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [1, 2, 3],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__min_samples_leaf": [1, 5, 10, 20],
    },
}