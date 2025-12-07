# For running this file in terminal: 
# cd /Users/mohammedmubashiruddinfaraz/Documents/Machine_Learning/Decision_Tree
# "./.venv/bin/python" "Tree.py"


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, subprocess
import joblib
from pathlib import Path
from retrain_tree import _find_dataset, _prep
AUTO_OPEN = True  
generated_files = []

dataset = pd.read_csv('stroke_data_preprocessed.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].astype(int).to_numpy() 
feature_names = dataset.columns[:-1].tolist()

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(X_train)
print(y_train)
print(X_test)
print(y_test)


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay,
                             PrecisionRecallDisplay, average_precision_score)


dt0 = DecisionTreeClassifier(
    random_state=42, class_weight="balanced", max_depth=5
).fit(X_train, y_train)

proba0 = dt0.predict_proba(X_test)[:,1]
pred0  = dt0.predict(X_test)
print("ROC-AUC (baseline):", roc_auc_score(y_test, proba0))
print(classification_report(y_test, pred0, digits=3, zero_division=0))

ConfusionMatrixDisplay(confusion_matrix(y_test, pred0)).plot()
plt.tight_layout(); plt.savefig("dt_confusion.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_confusion.png"))


RocCurveDisplay.from_estimator(dt0, X_test, y_test)
plt.tight_layout(); plt.savefig("dt_roc.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_roc.png"))
PrecisionRecallDisplay.from_estimator(dt0, X_test, y_test)
plt.tight_layout(); plt.savefig("dt_pr.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_pr.png"))
print("Average Precision (AUPRC):", average_precision_score(y_test, proba0))

imp = pd.Series(dt0.feature_importances_, index=feature_names).sort_values(ascending=False)
ax = imp.head(15).plot(kind="bar"); ax.set_ylabel("Importance")
plt.tight_layout(); plt.savefig("dt_feature_importance.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_feature_importance.png"))

plt.figure(figsize=(22,12))
plot_tree(dt0, feature_names=feature_names, class_names=['No Stroke','Stroke'],
          filled=True, rounded=True, impurity=False, proportion=True, max_depth=5)
plt.tight_layout(); plt.savefig("dt_tree.png", dpi=200); plt.close()
generated_files.append(os.path.abspath("dt_tree.png"))

param_grid = {
    "criterion": ["gini", "entropy"],  
    "max_depth": [3, 5, 7, 9, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 3, 5, 10],
    "max_features": [None, "sqrt", 0.5],
    "class_weight": [
        None,
        "balanced",
        {0: 1, 1: 3},  
        {0: 1, 1: 5}, 
    ],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=1,
    refit=True,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("Best params:", grid.best_params_)
proba = best.predict_proba(X_test)[:,1]
pred  = best.predict(X_test)

rows = []
for name, clf in [("DT_baseline", dt0), ("DT_tuned", best)]:
    p = clf.predict_proba(X_test)[:,1]
    yhat = (p>=0.5).astype(int)
    rpt = classification_report(y_test, yhat, labels=[0,1], output_dict=True, zero_division=0)
    rows.append([name, roc_auc_score(y_test, p),
                 rpt["1"]["precision"], rpt["1"]["recall"], rpt["1"]["f1-score"],
                 rpt["accuracy"]])
pd.DataFrame(rows, columns=["Model","ROC_AUC","Prec(1)","Rec(1)","F1(1)","Accuracy"])\
  .to_csv("dt_results_summary.csv", index=False)

path = DecisionTreeClassifier(random_state=42, class_weight="balanced").cost_complexity_pruning_path(X_train, y_train)
ccp_values = np.unique(path.ccp_alphas)
cv_scores = []
for a in ccp_values:
    clf = DecisionTreeClassifier(random_state=42, class_weight="balanced", ccp_alpha=a)
    scores = []
    for tr, va in cv.split(X_train, y_train):
        clf.fit(X_train[tr], y_train[tr])
        scores.append(roc_auc_score(y_train[va], clf.predict_proba(X_train[va])[:,1]))
    cv_scores.append(np.mean(scores))
best_alpha = ccp_values[int(np.argmax(cv_scores))]
dt_pruned = DecisionTreeClassifier(random_state=42, class_weight="balanced", ccp_alpha=best_alpha).fit(X_train, y_train)
print("Chosen ccp_alpha:", best_alpha, "ROC-AUC (pruned):", roc_auc_score(y_test, dt_pruned.predict_proba(X_test)[:,1]))

try:
    model_bundle = {
        "model": dt_pruned,
        "feature_names": feature_names, 
    }
    joblib.dump(model_bundle, "stroke_dt_model.pkl")
    print("Saved pruned model to stroke_dt_model.pkl")
except Exception as e:
    print("WARNING: could not save model to stroke_dt_model.pkl:", e)

from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, f1_score

prec, rec, thr = precision_recall_curve(y_test, proba) 

f1 = (2 * prec * rec) / (prec + rec + 1e-12)
f1_thr = thr[np.argmax(f1[:-1])]
yhat_f1 = (proba >= f1_thr).astype(int)
print(f"F1-optimal threshold: {f1_thr:.4f}")
print(classification_report(y_test, yhat_f1, digits=3, zero_division=0))

target_recall = 0.80
mask = rec[:-1] >= target_recall
if np.any(mask):
    cand_thr = thr[mask]
    cand_prec = prec[:-1][mask]
    idx = np.argmax(cand_prec)
    hi_thr = max(cand_thr[idx], 1e-6) 
else:
    hi_thr = f1_thr 
yhat_hi = (proba >= hi_thr).astype(int)
print(f"High-recall threshold (~{target_recall:.0%} recall): {hi_thr:.4f}")
print(classification_report(y_test, yhat_hi, digits=3, zero_division=0))

fpr, tpr, roc_thr = roc_curve(y_test, proba)
youden_idx = np.argmax(tpr - fpr)
youd_thr = roc_thr[youden_idx]
yhat_youd = (proba >= youd_thr).astype(int)
print(f"Youden J threshold: {youd_thr:.4f}")
print(classification_report(y_test, yhat_youd, digits=3, zero_division=0))

if generated_files:
    print("\nGenerated figures:")
    for f in generated_files:
        print(" -", f)
        if AUTO_OPEN and os.path.exists(f):
            try:
                subprocess.run(["/usr/bin/open", "-a", "Preview", f], check=False)
            except Exception:
                try:
                    subprocess.run(["/usr/bin/open", f], check=False)
                except Exception:
                    pass
else:
    print("\n(No figures recorded. Check save paths.)")