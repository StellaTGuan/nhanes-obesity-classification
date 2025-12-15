# ============================================================
# 02_model_logistic.R
# Multinomial logistic regression (3-class BMI outcome)
# + simple penalization diagnostics (+ optional L1 robustness)
# ============================================================

# -------------------------
# Packages
# -------------------------
required_pkgs <- c(
  "tidyverse", "readr", "caret", "nnet", "yardstick", "glmnet", "Matrix"
)

installed <- rownames(installed.packages())
to_install <- setdiff(required_pkgs, installed)
if (length(to_install) > 0) install.packages(to_install)

library(tidyverse)
library(readr)
library(caret)
library(nnet)
library(yardstick)

set.seed(1)


# -------------------------
# Load data
# -------------------------
path <- "data/processed/nhanes_adults20_bmi_features.csv"
dat <- read_csv(path, show_col_types = FALSE)

# -------------------------
# Outcome: 3-class BMI (Underweight+Normal combined)
# NOTE: Use safe factor levels for caret classProbs.
# -------------------------
dat <- dat %>%
  mutate(
    bmi_cat3 = case_when(
      bmi < 25 ~ "UnderNormal",
      bmi < 30 ~ "Overweight",
      TRUE     ~ "Obese"
    ),
    bmi_cat3 = factor(bmi_cat3, levels = c("UnderNormal", "Overweight", "Obese"))
  )

# -------------------------
# Train/test split (stratified)
# -------------------------
idx <- createDataPartition(dat$bmi_cat3, p = 0.75, list = FALSE)
train <- dat[idx, ]
test  <- dat[-idx, ]

# -------------------------
# Missing handling AFTER split (avoid leakage)
# - Factors: NA -> "Unknown"
# - Numeric: add *_na flags + median impute using TRAIN medians
# -------------------------
factor_cols <- c("sex", "educ", "marital")

# NEW dietary predictors added to modeling set
num_cols <- c(
  "age", "pir", "alcohol_drinks_per_day", "sleep_hours_avg", "mvpa_eq_min_wk",
  "kcal_day1", "fiber_day1", "satfat_day1"
)

make_unknown_factor <- function(x) {
  x <- as.character(x)
  x[is.na(x) | x == ""] <- "Unknown"
  factor(x)
}

train <- train %>% mutate(across(all_of(factor_cols), make_unknown_factor))
test  <- test  %>% mutate(across(all_of(factor_cols), make_unknown_factor))

train_medians <- sapply(train[num_cols], function(v) median(v, na.rm = TRUE))

for (v in num_cols) {
  flag <- paste0(v, "_na")
  train[[flag]] <- is.na(train[[v]])
  test[[flag]]  <- is.na(test[[v]])
  
  train[[v]] <- if_else(is.na(train[[v]]), train_medians[[v]], train[[v]])
  test[[v]]  <- if_else(is.na(test[[v]]),  train_medians[[v]], test[[v]])
}

# Keep only modeling columns (don’t use id or raw bmi)
keep_vars <- c("bmi_cat3", factor_cols, num_cols, paste0(num_cols, "_na"))
train_mod <- train %>% select(all_of(keep_vars))
test_mod  <- test  %>% select(all_of(keep_vars))

# -------------------------
# 3.1 Baseline multinomial logistic regression (caret + nnet::multinom)
# - Tune decay with 5-fold CV
# - Center/scale predictors
# -------------------------
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE
)

grid <- expand.grid(decay = c(0, 1e-4, 1e-3, 1e-2, 1e-1))

fit_multinom <- train(
  bmi_cat3 ~ .,
  data = train_mod,
  method = "multinom",
  trControl = ctrl,
  tuneGrid = grid,
  preProcess = c("center", "scale"),
  metric = "Accuracy"
)

# Save tuning results (useful for report + reproducibility)
write_csv(fit_multinom$results, "output/tables/multinom_cv_results.csv")

# -------------------------
# Test evaluation (baseline)
# -------------------------
pred_class <- predict(fit_multinom, newdata = test_mod)

cm <- caret::confusionMatrix(pred_class, test_mod$bmi_cat3)
sink("output/tables/multinom_confusion_matrix.txt")
print(cm)
sink()

test_eval <- tibble(truth = test_mod$bmi_cat3, estimate = pred_class)

metrics_tbl <- tibble(
  accuracy = yardstick::accuracy_vec(test_eval$truth, test_eval$estimate),
  macro_f1 = yardstick::f_meas_vec(test_eval$truth, test_eval$estimate, estimator = "macro")
)
write_csv(metrics_tbl, "output/tables/multinom_test_metrics.csv")

# Confusion matrix figure (one figure is enough for report)
cm_df <- as.data.frame(cm$table)
p_cm <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 4) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix (Multinomial Logit)", x = "True class", y = "Predicted class") +
  theme_minimal(base_size = 13) +
  theme(panel.grid = element_blank())

ggsave("output/figures/multinom_confusion_matrix.png", p_cm, width = 6.5, height = 5, dpi = 300)

# -------------------------
# 3.1.2 Penalization diagnostics
# (A) CV vs test gap (overfitting signal)
# (B) Max absolute coefficient (instability/separation signal)
# -------------------------
best_cv_acc <- max(fit_multinom$results$Accuracy, na.rm = TRUE)
test_acc <- metrics_tbl$accuracy[1]

gap_tbl <- tibble(
  best_cv_accuracy = best_cv_acc,
  test_accuracy    = test_acc,
  cv_minus_test    = best_cv_acc - test_acc
)
write_csv(gap_tbl, "output/tables/multinom_cv_vs_test_gap.csv")

raw_model <- fit_multinom$finalModel
coefs_mat <- coef(raw_model)
max_abs_coef <- max(abs(coefs_mat), na.rm = TRUE)

coef_diag <- tibble(
  max_abs_coef = max_abs_coef
)
write_csv(coef_diag, "output/tables/multinom_coef_diagnostic.csv")

# -------------------------
# Quick console output
# -------------------------
cat("\n=== Multinomial logit (baseline) ===\n")
print(fit_multinom)
cat("\nTest metrics:\n")
print(metrics_tbl)
cat("\nDiagnostics:\n")
print(gap_tbl)
print(coef_diag)

library(dplyr)
library(tibble)
library(purrr)

coefs <- coef(fit_multinom$finalModel)

coef_tbl <- purrr::imap_dfr(
  asplit(coefs, 1),  # split by row (class)
  ~ tibble(
    class = .y,                      # e.g., "Overweight", "Obese"
    term  = names(.x),
    beta  = as.numeric(.x),
    odds_ratio = exp(beta)
  )
) %>%
  arrange(class, desc(abs(beta)))

coef_tbl

coef_tbl %>%
  group_by(class) %>%
  slice_max(order_by = abs(beta), n = 10) %>%
  ungroup()

write_csv(coef_tbl, "output/tables/multinom_coefs.csv")
