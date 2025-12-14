# nhanes-obesity-classification
Econ 695 project: predicting adult BMI category from NHANES 2021–2023 using logistic regression and boosting

## File Structure
```text
nhanes-obesity-classification/
├─ README.md
├─ nhanes-obesity.Rproj
├─ data/
│  ├─ raw/
│  │  ├─ BMX_L.xpt
│  │  ├─ DEMO_L.xpt
│  │  ├─ SLQ_L.xpt
│  │  ├─ PAQ_L.xpt
│  │  └─ ALQ_L.xpt
│  └─ processed/
│     └─ nhanes_adults20_bmi_features.csv
├─ R/
│  ├─ 01_clean_nhanes.R
│  ├─ 02_model_logistic.R
│  ├─ 03_model_boosting.R
│  └─ 04_evaluate_plots.R
├─ project_report.pdf
├─ output/
│  ├─ figures/
│  └─ tables/
└─ .gitignore
