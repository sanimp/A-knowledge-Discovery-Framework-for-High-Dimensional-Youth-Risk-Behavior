# A Knowledge Discovery Framework for High-Dimensional Youth Risk Behavior
## From Exploratory Data Analysis to Predictive Modelling

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)

---

## Project Overview

This project presents a comprehensive **Knowledge Discovery and Data Mining (KDD)** framework applied to the **CDC Youth Risk Behavior Surveillance System (YRBS)** dataset — a nationally representative longitudinal survey of adolescent risk behaviors spanning **1991 to 2017**, comprising over **1.17 million records** across 31 variables.

The pipeline covers the full spectrum of data science practice: from raw data ingestion and quality assessment, through exploratory data analysis and statistical inference, to ensemble machine learning with interpretable SHAP-based feature analysis.

---

## Authors & Collaborators

| Name | Institution | Email |
|---|---|---|
| **Priscilla Sarfoa Anim** | Grand Valley State University | animp@mail.gvsu.edu |
| **Victor Entsie** | Grand Valley State University | entsiev@mail.gvsu.edu |
| **Kwame Nyankyerah** | Grand Valley State University | nyankyek@mail.gvsu.edu |
| **Esther Djan** | Grand Valley State University | Djane@mail.gvsu.edu |

---

## Research Questions

1. How have youth risk behavior prevalence rates changed over time across different demographic groups and geographic regions?
2. Are there statistically significant differences in risk behavior rates by sex and grade level?
3. Can machine learning models accurately predict risk behavior prevalence, and which features are most predictive?

---

## Repository Structure

```
├── notebook/
│   └── KDD_Project_Analysis.ipynb                  # Main analysis notebook
├── data/
│   └── README.md                                    # Data download instructions
├── figures/
│   ├── column_distributions.png
│   ├── risk_over_time.png
│   ├── correlation_heatmap.png
│   ├── sex_analysis.png
│   ├── grade_analysis.png
│   ├── usa_risk_map_fallback.png
│   ├── usa_risk_map_animated.html
│   ├── state_comparison.png
│   ├── top_behaviors.png
│   └── roc_curve.png
├── report/
│   ├── main.tex                                     # KDD-format LaTeX report
│   └── sources.bib                                  # BibTeX references
└── README.md
```

---

## Dataset

**Source:** CDC Youth Risk Behavior Surveillance System (YRBS)
**Provided by:** [Child Health and Community Wellness (CHC)](https://www.chconline.org/resourcelibrary/youth-risk-behavior-survey-data-summary-trends-report-2013-2023-downloadable/)

| Property | Value |
|---|---|
| Records | 1,176,120 |
| Variables | 31 |
| Years Covered | 1991 – 2017 |
| Survey Type | Biennial, nationally representative |
| Target Variable | `Greater_Risk_Data_Value` (%) |

> **Note:** The dataset file is not included in this repository due to its size. Please download it directly from the CHC link above and place it in the `data/` folder before running the notebook.

---

## Methodology

### Data Preprocessing
- Dropped 4 columns with 100% missing values
- Excluded confidence limit fields (`Greater_Risk_Low_Confidence_Limit`, `Greater_Risk_High_Confidence_Limit`) and `Lesser_Risk_Data_Value` to prevent data leakage
- Converted all object columns to categorical `dtype`
- Applied median imputation for numerical features
- One-hot encoding for categorical features within scikit-learn `Pipeline` objects
- Retained 353,522 complete cases after dropping missing target rows
- Used a stratified 100,000-record sample for model training to ensure computational tractability

### Exploratory Data Analysis
- Univariate distribution analysis
- Temporal trend analysis (1991–2017)
- Geographic risk profiling across 46 US states and territories with interactive Plotly choropleth maps
- Demographic subgroup analysis by sex and grade with 95% confidence intervals
- Behavioral correlation heatmap across risk categories

### Feature Selection
Features were selected based on genuine predictive relevance with no data leakage:

```
YEAR · LocationAbbr · Topic · Subtopic · Sex · Race · Grade · Data_Value_Type
```

### Machine Learning Models
Both **regression** (predicting exact risk %) and **classification** (high vs. low risk, thresholded at median 11.35%) tasks were performed using:

| Model | Type |
|---|---|
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |
| XGBoost | Gradient Boosted Trees |
| CatBoost | Gradient Boosted Trees |

- 80/20 stratified train-test split (`random_state=42`)
- `GridSearchCV` hyperparameter tuning with 5-fold cross-validation
- Best XGBoost parameters: `n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8`
- 5-fold cross-validation for generalization assessment
- SHAP `TreeExplainer` for model interpretability

---

## Key Results

### Model Performance

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

### Summary of Key Findings

| Metric | Value |
|---|---|
| Best Regression R² | 0.4641 (CatBoost) |
| Best Classification Accuracy | 71.95% (XGBoost) |
| Best AUC-ROC | 0.7813 (XGBoost) |
| Highest Risk Location | Denver, CO (33.01%) |
| Highest Risk Behavior | Ever alcohol use (68.49%) |
| Male vs Female Risk Gap | 19.99% vs 17.39% (significant at α = 0.05) |
| Overall Risk Trend | Declining from ~21% (1995) to ~17% (2015) |

---

## How to Run

### Google Colab 
1. Open the notebook in Google Colab
2. Upload the dataset to your Google Drive
3. Update the file path in Cell 3:
```python
file_path = "/content/Alcohol and Other Drug Use.csv.zip"
```
4. Run all cells from top to bottom (`Runtime → Run all`)

> **Note:** Cell 2 installs `catboost` and `shap` automatically. The full ML pipeline including GridSearchCV may take 10–20 minutes to complete.

2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost shap scipy plotly
```

3. Launch Jupyter and open the notebook:
```bash
jupyter notebook notebook/KDD_Project_Analysis.ipynb
```

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
catboost
shap
scipy
plotly
jupyter
```

---

## References

- Yard, E., et al. (2021). Emergency Department Visits for Suspected Suicide Attempts Among Persons Aged 12–25 Years. *Morbidity and Mortality Weekly Report*. https://pmc.ncbi.nlm.nih.gov/articles/PMC10156160/
- Tompkins, N. O. (2023). Youth Risk Behavior Analysis: Epidemiological Trends and Interventions. *ERIC*. https://eric.ed.gov/?id=ED674601
- Child Health and Community Wellness. (2023). *YRBS Data Summary and Trends Report: 2013–2023*. https://www.chconline.org/resourcelibrary/youth-risk-behavior-survey-data-summary-trends-report-2013-2023-downloadable/
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Prokhorenkova, L., et al. (2018). CatBoost: Unbiased Boosting with Categorical Features. *NeurIPS*.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

We gratefully acknowledge the **Centers for Disease Control and Prevention (CDC)** for making the YRBS dataset publicly accessible, and the **Child Health and Community Wellness** organization for the curated data summary report. All analyses were conducted using open-source Python libraries including scikit-learn, XGBoost, CatBoost, and SHAP.
