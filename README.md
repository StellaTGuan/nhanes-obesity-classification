# nhanes-obesity-classification
Econ 695 project: predicting adult BMI category from NHANES 2021–2023 using logistic regression and boosting

## File Structure
```text
nhanes-obesity-classification/
├─ README.md
├─ nhanes-obesity.Rproj          # RStudio project (optional)
├─ data/
│  ├─ raw/                       # .gitignored (NHANES XPT/CSV files)
│  └─ processed/                 # small, cleaned CSV/RDS (maybe tracked)
├─ R/
│  ├─ 01_download_nhanes.R
│  ├─ 02_clean_nhanes.R
│  ├─ 03_model_logistic.R
│  ├─ 04_model_boosting.R
│  └─ 05_evaluate_plots.R
├─ analysis/
│  └─ nhanes_obesity_report.Rmd  # main RMarkdown report
├─ output/
│  ├─ figures/                   # plots for the report
│  └─ tables/                    # model summary tables, metrics, etc.
└─ .gitignore
