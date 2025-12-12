# Stroke Prediction Model 🧠📊

This repository contains an end-to-end **Stroke Risk Prediction** system built using Machine Learning.  
It includes model training + retraining scripts, a saved trained model (`.pkl`), evaluation artifacts (ROC/PR curves, confusion matrix, feature importance), and a small web app for live predictions.

---

## Project Overview

Stroke is a serious medical condition where early risk screening can help with prevention and timely intervention.  
This project trains a supervised ML model to estimate stroke risk from input health/lifestyle features.

### What’s included
- Model training using a **Decision Tree** (`Tree.py`)
- Model retraining / improvement workflow (`retrain_tree.py`)
- Saved trained model file (`stroke_dt_model.pkl`)
- Evaluation outputs:
  - Confusion Matrix
  - ROC Curve
  - Precision–Recall (PR) Curve
  - Feature Importance plot
  - CSV summary of results
- A front-end web app for running predictions (`stroke-risk-prediction-app/`)

---

## Tech Stack

- **Python**
- **scikit-learn** (Decision Tree, metrics)
- **pandas / numpy**
- **Matplotlib**
- **Web App:** JavaScript/TypeScript (Next.js-style app folder)

---

## Repository Structure (High Level)

Stroke-Prediction-Model/
├─ Tree.py                         # Train baseline Decision Tree model
├─ retrain_tree.py                 # Retrain / improve model (tuning / balancing)
├─ stroke_api.py                   # API / inference helper (if used)
├─ stroke_dt_model.pkl             # Saved trained model (pickle)
├─ stroke_data_preprocessed.csv    # Preprocessed dataset
├─ dt_confusion.png                # Confusion matrix plot
├─ dt_roc.png                      # ROC curve plot
├─ dt_pr.png                       # Precision–Recall curve plot
├─ dt_feature_importance.png       # Feature importance plot
├─ dt_results_summary.csv          # Metrics summary
└─ stroke-risk-prediction-app/     # Web app for live predictions

---

## How to Run (Model Training)

### 1) Create environment and install dependencies
bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn matplotlib

2) Train the baseline model
python Tree.py

3) Retrain / improve the model
python retrain_tree.py
After running, check the generated plots (dt_roc.png, dt_pr.png, etc.) and the results summary CSV.

How to Run (Web App)
cd stroke-risk-prediction-app
npm install
npm run dev

Open:
	•	http://localhost:3000


Results (Artifacts)

This repo contains saved evaluation visuals and summaries:
•	ROC and PR curves for classification performance
•	Confusion matrix for error analysis
•	Feature importance to interpret which inputs drive predictions

⸻

Notes / Future Improvements
•	Try stronger models (Random Forest, XGBoost)
•	Add calibration for better probability estimates
•	Add cross-validation and stronger hyperparameter search
•	Improve the UI and connect the web app to a backend inference API

⸻

Author

Mohammed Mubashir Uddin Faraz
GitHub: https://github.com/machackgo
