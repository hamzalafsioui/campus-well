
from datetime import datetime, timedelta
import sys
import os

from airflow import DAG
from airflow.operators.python import PythonOperator

# ----- default arguments 
default_args = {
    "owner": "campuswell",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ----- make sure /workspace is on the Python path 
WORKSPACE = "/workspace"
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)


# ========================================================
# TASK 1 | Load raw data, clean it, split into train / test
# ========================================================
def task_prepare_data():
    import pandas as pd
    from src import config
    from src.data import load_raw, clean_data, split_data

    print("=> Loading raw data …")
    df = load_raw()
    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns")

    print("=> Cleaning data …")
    df = clean_data(df)

    print("=> Splitting into train / test …")
    train, test, info = split_data(df, save=True)
    print(f"  Train: {len(train)} rows | Test: {len(test)} rows")
    print(f"  Missing-target rows dropped : {info['n_dropped_missing_target']}")
    print(f"  Duplicate rows dropped      : {info['n_dropped_duplicates']}")
    print(f"  Train distribution          : {info['train_distribution']}")


# ========================================================
# TASK 2 | Train several models with 5-fold CV & save a comparison report
# ========================================================
def task_train_models():
    import json
    import pandas as pd
    from sklearn.model_selection import cross_validate

    from src import config
    from src.preprocessing import build_model_pipeline, prepare_features, prepare_target
    from src.models import get_models, get_cv, SCORING

    print("=> Reading training data …")
    train = pd.read_csv(config.DATA_INTERIM / "train.csv")

    X = prepare_features(train)
    y = prepare_target(train)

    cv = get_cv()
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"  Cross-validating: {name} …")
        pipeline = build_model_pipeline(model)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1)

        results[name] = {
            metric: round(float(scores[f"test_{metric}"].mean()), 4)
            for metric in SCORING
        }
        print(f"    accuracy={results[name]['accuracy']}  "
              f"f1_macro={results[name]['f1_macro']}  "
              f"roc_auc={results[name].get('roc_auc_ovr', 'n/a')}")

    # save comparison
    out_path = config.MODELS / "cv_comparison.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"=> CV comparison saved -> {out_path}")


# ========================================================
# TASK 3 | GridSearchCV on the best two models, pick winner
# ========================================================
def task_optimize_model():
    import json
    import pandas as pd
    from sklearn.model_selection import GridSearchCV

    from src import config
    from src.preprocessing import build_model_pipeline, prepare_features, prepare_target
    from src.models import get_models, get_cv, PARAM_GRIDS

    print("=> Reading training data …")
    train = pd.read_csv(config.DATA_INTERIM / "train.csv")
    X = prepare_features(train)
    y = prepare_target(train)

    # read CV results to decide which models to tune
    cv_path = config.MODELS / "cv_comparison.json"
    cv_results = json.loads(cv_path.read_text())

    # pick models that have a param grid defined
    models_to_tune = {
        name: model
        for name, model in get_models().items()
        if name in PARAM_GRIDS
    }

    best_score = -1
    best_name = None
    best_estimator = None
    best_params = None

    for name, model in models_to_tune.items():
        print(f"=> GridSearchCV: {name} …")
        pipeline = build_model_pipeline(model)
        gs = GridSearchCV(
            pipeline,
            param_grid=PARAM_GRIDS[name],
            cv=get_cv(),
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X, y)
        score = round(gs.best_score_, 4)
        print(f"  Best f1_macro={score}  params={gs.best_params_}")

        if score > best_score:
            best_score = score
            best_name = name
            best_estimator = gs.best_estimator_
            best_params = gs.best_params_

    # save winner info
    winner = {"name": best_name, "best_params": best_params, "best_cv_f1": best_score}
    winner_path = config.MODELS / "best_model_info.json"
    winner_path.write_text(json.dumps(winner, indent=2), encoding="utf-8")
    print(f"=> Winner: {best_name} (f1_macro={best_score})")

    # save fitted pipeline temporarily so evaluate task can load it
    import joblib
    tmp_path = config.MODELS / "best_pipeline_tmp.joblib"
    joblib.dump(best_estimator, tmp_path)
    print(f"=> Tmp pipeline saved -> {tmp_path}")


# ========================================================
# TASK 4 | Evaluate optimised model on the held-out test set
# ========================================================
def task_evaluate_model():
    import json
    import joblib
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    from src import config
    from src.preprocessing import prepare_features, prepare_target
    from src.evaluation import compute_metrics, save_json

    print("=> Loading test data …")
    test = pd.read_csv(config.DATA_INTERIM / "test.csv")
    X_test = prepare_features(test)
    y_test = prepare_target(test)

    print("=> Loading best pipeline …")
    pipeline = joblib.load(config.MODELS / "best_pipeline_tmp.joblib")

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    classes = list(pipeline.classes_)

    print("=> Computing metrics …")
    metrics = compute_metrics(y_test, y_pred, y_proba, classes)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    # read winner info
    winner = json.loads((config.MODELS / "best_model_info.json").read_text())

    result = {
        "model_name"  : f"{winner['name']} (tuned)",
        "best_params" : winner["best_params"],
        "features"    : list(X_test.columns),
        "classes"     : classes,
        "n_test"      : len(y_test),
        "cv_f1_macro" : winner["best_cv_f1"],
        "test_metrics": metrics,
        "classification_report": report,
        "confusion_matrix"     : cm,
        "run_at": datetime.utcnow().isoformat(),
        "warning": "Pedagogical prototype. Not a diagnostic tool.",
    }

    out = config.MODELS / "evaluation.json"
    save_json(result, out)
    print(f"=> Evaluation saved → {out}")


# ========================================================
# TASK 5 | Save the final pipeline as final_pipeline.joblib
# ========================================================
def task_save_pipeline():
    import joblib
    from src import config

    tmp  = config.MODELS / "best_pipeline_tmp.joblib"
    final = config.MODELS / "final_pipeline.joblib"

    pipeline = joblib.load(tmp)
    joblib.dump(pipeline, final)
    print(f"=> Final pipeline saved -> {final}")

    # clean up tmp file
    tmp.unlink(missing_ok=True)
    print("=> Tmp file removed.")


# ========================================================
# DAG DEFINITION
# ========================================================
with DAG(
    dag_id="campuswell_pipeline",
    default_args=default_args,
    description="CampusWell | Full ML pipeline (prepare -> train -> optimize -> evaluate -> save)",
    schedule_interval="0 2 * * *",   # every day at 02:00 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    is_paused_upon_creation=True,
    tags=["campuswell", "ml", "pipeline"],
) as dag:

    t1 = PythonOperator(
        task_id="prepare_data",
        python_callable=task_prepare_data,
    )

    t2 = PythonOperator(
        task_id="train_models",
        python_callable=task_train_models,
    )

    t3 = PythonOperator(
        task_id="optimize_model",
        python_callable=task_optimize_model,
    )

    t4 = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate_model,
    )

    t5 = PythonOperator(
        task_id="save_pipeline",
        python_callable=task_save_pipeline,
    )

    # chain
    t1 >> t2 >> t3 >> t4 >> t5
