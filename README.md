# A Knowledge Discovery Framework for High-Dimensional Youth Risk Behavior
### From Exploratory Data Analysis to Predictive Modelling

> **KDD '26 — Grand Valley State University**
> Priscilla Sarfoa Anim · Victor Entsie · Kwame Nyankyerah · Esther Djan

---

## Overview

This project presents a comprehensive knowledge discovery and data mining (KDD) framework applied to the **CDC Youth Risk Behavior Surveillance System (YRBS)** dataset — a nationally representative longitudinal survey spanning **1991 to 2017**, comprising over **1.17 million records** across 31 variables.

The framework covers the full analytical pipeline:
- Data preprocessing and quality assessment
- Exploratory data analysis (EDA)
- Statistical inference via confidence interval estimation
- Geographic risk profiling with interactive maps
- Ensemble machine learning for regression and classification
- Model interpretability via SHAP analysis

---

## Dataset

| Property | Value |
|---|---|
| Source | CDC Youth Risk Behavior Surveillance System (YRBS) |
| Records | 1,176,120 |
| Variables | 31 |
| Survey Years | 1991 – 2017 |
| Target Variable | `Greater_Risk_Data_Value` (% of respondents reporting greater-risk behavior) |

The dataset is sourced from the [Child Health and Community Wellness resource library](https://www.chconline.org/resourcelibrary/youth-risk-behavior-survey-data-summary-trends-report-2013-2023-downloadable/).

To load the dataset in the notebook, upload the file to your Google Drive and update the file path:
```python
file_path = "/content/Alcohol and Other Drug Use.csv.zip"
df = pd.read_csv(file_path, compression='zip')
```

---

## Key Findings

### Descriptive Analysis
- Mean risk behavior prevalence: **18.76%** (SD = 19.03%)
- Overall declining trend from ~21% in 1995 to ~17% in 2015
- **Lifetime alcohol use** is the highest-risk behavior (mean 68.49%)

### Demographic Disparities
| Group | Mean Risk (%) | CI Lower | CI Upper |
|---|---|---|---|
| Female | 17.39 | 17.28 | 17.51 |
| Male | 19.99 | 19.88 | 20.11 |

Males exhibit significantly higher risk prevalence than females (non-overlapping 95% CIs, α = 0.05). Risk prevalence also increases progressively from 9th through 12th grade.

### Geographic Profiling
- **Highest risk:** Denver, CO (33.01%), Cleveland, OH (26.95%), Palau (24.73%)
- **Lowest risk:** Virginia, Utah, Pennsylvania

### Predictive Modelling Results

**Regression** (predicting exact risk prevalence %):

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **CatBoost** | **0.4641** | **14.11** | **10.94** |
| XGBoost | 0.4605 | 14.15 | 10.96 |
| Gradient Boosting | 0.4428 | 14.39 | 11.17 |
| Random Forest | 0.3267 | 15.81 | 11.68 |

**Classification** (predicting High vs Low risk group):

| Model | Accuracy | AUC-ROC |
|---|---|---|
| **XGBoost** | **0.7195** | **0.7813** |
| CatBoost | 0.7175 | 0.7798 |
| Gradient Boosting | 0.7114 | 0.7761 |
| Random Forest | 0.6794 | 0.7487 |

---

## Project Structure

```
 project
│
├── KDD_Project_Analysis.ipynb     # Main notebook (run in Google Colab)
├── README.md                      # This file
│
└── outputs/                       # Generated figures and maps
    ├── risk_over_time.png
    ├── usa_risk_map_plotly.html   # Interactive choropleth map
    ├── usa_risk_map_animated.html # Animated map by year
    ├── sex_analysis.png
    ├── grade_analysis.png
    ├── top_behaviors.png
    ├── correlation_heatmap.png
    └── roc_curve.png
```

---

## Notebook Structure

| Section | Description |
|---|---|
| **1 — Importing Libraries** | All dependencies installed and imported |
| **2 — Loading the Dataset** | Google Drive mount and data loading |
| **3 — Initial Exploration** | Head, dtypes, shape |
| **4 — Data Quality Assessment** | Missing values, duplicates |
| **5 — Descriptive Statistics** | Summary stats, distributions |
| **6 — Research Questions** | Temporal trends, state rankings, correlations |
| **7 — Confidence Interval Analysis** | Sex and grade subgroup analysis |
| **8 — Geographic Analysis** | Interactive and static US maps |
| **9 — Correlation Analysis** | Behavioral correlation matrix |
| **10 — Machine Learning** | Feature selection, preprocessing, model training |
| **11 — Model Evaluation** | Regression residuals, ROC curve, confusion matrix |
| **12 — SHAP Analysis** | Feature importance and interpretability |

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `KDD_Project_Analysis.ipynb` via **File → Upload notebook**
3. Upload the dataset (`Alcohol and Other Drug Use.csv.zip`) to your Google Drive
4. Update the file path in Cell 3 to match your Drive location
5. Run all cells: **Runtime → Run all**

> **Note:** The notebook installs `catboost` and `shap` automatically in Cell 2. Runtime may take 10–20 minutes for the full ML pipeline due to GridSearchCV hyperparameter tuning.

---

## Dependencies

All dependencies are installed automatically in Cell 2. Key libraries:

```python
pandas · numpy · matplotlib · seaborn · scipy
scikit-learn · xgboost · catboost · shap · plotly
```

---

## Methodology

### Feature Selection
Features were carefully selected to avoid **data leakage**. Confidence limit fields (`Greater_Risk_Low_Confidence_Limit`, `Greater_Risk_High_Confidence_Limit`) and the complementary outcome field (`Lesser_Risk_Data_Value`) were excluded as they are statistically derived from the target variable. The final feature set used:

```
YEAR · LocationAbbr · Topic · Subtopic · Sex · Race · Grade · Data_Value_Type
```

### Preprocessing Pipeline
- Missing target rows dropped → 353,522 complete cases retained
- 100,000 record stratified sample drawn for model training
- Categorical features: one-hot encoded (unknown category handling)
- Numerical features: median imputation
- All steps embedded in `scikit-learn Pipeline` objects

### Hyperparameter Tuning
XGBoost regressor tuned via `GridSearchCV` (5-fold CV):
```
n_estimators ∈ {100, 200}
max_depth ∈ {4, 6, 8}
learning_rate ∈ {0.05, 0.1}
subsample ∈ {0.8, 1.0}
```
Best parameters: `n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8`

---

## Authors

| Name | Institution | Email |
|---|---|---|
| Priscilla Sarfoa Anim | Grand Valley State University | animp@mail.gvsu.edu |
| Victor Entsie | Grand Valley State University | entsiev@mail.gvsu.edu |
| Kwame Nyankyerah | Grand Valley State University | nyankyek@mail.gvsu.edu |
| Esther Djan | Grand Valley State University | Djane@mail.gvsu.edu |

---

## Acknowledgements

We gratefully acknowledge the **Centers for Disease Control and Prevention (CDC)** for making the YRBS dataset publicly accessible, and the **Child Health and Community Wellness** organization for the curated data summary report.

---

## References

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
3. Child Health and Community Wellness. (2023). *Youth Risk Behavior Survey Data Summary and Trends Report: 2013–2023*.
4. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
5. Prokhorenkova, L. et al. (2018). CatBoost: Unbiased Boosting with Categorical Features. *NeurIPS*.
6. Tompkins, N. O. et al. (2023). *Youth Risk Behavior Analysis: Epidemiological Trends and Interventions*. ERIC.
7. Yard, E. et al. (2021). Emergency Department Visits for Suspected Suicide Attempts Among Persons Aged 12–25. *MMWR*, 70(24), 888–894.
