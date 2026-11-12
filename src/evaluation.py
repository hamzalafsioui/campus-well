import json
from sklearn.metrics import (accuracy_score,f1_score,precision_score,recall_score,roc_auc_score)



def compute_metrics(y_true,y_pred,y_proba,classes):

    metrics = {
        "accuracy":accuracy_score(y_true,y_pred),
        "precision_macro":precision_score(y_true,y_pred,average="macro"),
        "recall_macro":recall_score(y_true,y_pred,average="macro"),
        "f1_macro":f1_score(y_true,y_pred,average="macro"),
        "f1_weighted": f1_score(y_true,y_pred,average="weighted"),
        "roc_auc_ovr":roc_auc_score(y_true,y_proba,multi_class="ovr",labels=classes),
    }



    return {name:round(float(value),4) for name,value in metrics.items()}



def save_json(data,path):

    def convert(value):
        if hasattr(value,"item"):
            return value.item()
        return str(value)

    path.write_text(json.dumps(data,indent=2,ensure_ascii = False,default=convert),encoding="utf-8")