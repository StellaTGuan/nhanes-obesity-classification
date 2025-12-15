# ============================================================
# 02_model_logistic.R
# Multinomial logistic regression (3-class BMI outcome)
# ASSUMES cleaned dataset has NO NAs (complete cases)
# ============================================================

required_pkgs <- c("tidyverse", "readr", "caret", "nnet", "yardstick")
installed <- rownames(installed.packages())
to_install <- setdiff(required_pkgs, installed)
if (length(to_install) > 0) install.packages(to_install)

library(tidyverse)
library(readr)
library(caret)
library(nnet)
library(yardstick)

set.seed(1)

dir.create("output/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)

# -------------------------
# Load data (complete cases)
# -------------------------
path <- "data/processed/nhanes_adults20_bmi_features.csv"
dat <- read_csv(path, show_col_types = FALSE)

# Outcome
dat <- dat %>%
  mutate(
    bmi_cat3 = case_when(
      bmi < 25 ~ "UnderNormal",
      bmi < 30 ~ "Overweight",
      TRUE     ~ "Obese"
    ),
    bmi_cat3 = factor(bmi_cat3, levels = c("UnderNormal", "Overweight", "Obese"))
  )

# Predictors used
factor_cols <- c("sex", "educ", "marital")
num_cols <- c(
  "age", "pir", "alcohol_drinks_per_day", "sleep_hours_avg", "mvpa_eq_min_wk",
  "kcal_day1", "fiber_day1", "satfat_day1"
)

keep_vars <- c("bmi_cat3", factor_cols, num_cols)

dat <- dat %>%
  select(any_of(keep_vars)) %>%
  mutate(across(all_of(factor_cols), ~ factor(.x)))

# Sanity check: should be NA-free
if (anyNA(dat)) stop("Found NA values in modeling dataset. Re-run 01_clean_nhanes.R.")

# -------------------------
# Train/test split (stratified)
# -------------------------
idx <- createDataPartition(dat$bmi_cat3, p = 0.75, list = FALSE)
train_mod <- dat[idx, ]
test_mod  <- dat[-idx, ]

# -------------------------
# Multinomial logistic regression (caret + nnet::multinom)
# -------------------------
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE)
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

write_csv(fit_multinom$results, "output/tables/multinom_cv_results.csv")

# -------------------------
# Test evaluation
# -------------------------
pred_class <- predict(fit_multinom, newdata = test_mod)

cm <- caret::confusionMatrix(pred_class, test_mod$bmi_cat3)
sink("output/tables/multinom_confusion_matrix.txt")
print(cm)
sink()

test_eval <- tibble(truth = test_mod$bmi_cat3, estimate = pred_class)
metrics_tbl <- tibble(
  accuracy = yardstick::accuracy_vec(test_eval$truth, test_eval$estimate),
  macro_f1  = yardstick::f_meas_vec(test_eval$truth, test_eval$estimate, estimator = "macro")
)
write_csv(metrics_tbl, "output/tables/multinom_test_metrics.csv")

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
# Diagnostics
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

coef_diag <- tibble(max_abs_coef = max_abs_coef)
write_csv(coef_diag, "output/tables/multinom_coef_diagnostic.csv")

# Optional: top coefficients table
library(purrr)
coefs <- coef(fit_multinom$finalModel)
coef_tbl <- purrr::imap_dfr(
  asplit(coefs, 1),
  ~ tibble(class = .y, term = names(.x), beta = as.numeric(.x), odds_ratio = exp(beta))
) %>%
  arrange(class, desc(abs(beta)))

coef_tbl %>%
  group_by(class) %>%
  slice_max(order_by = abs(beta), n = 10) %>%
  ungroup()

cat("\n=== Multinomial logit (complete cases) ===\n")
print(fit_multinom)
cat("\nTest metrics:\n")
print(metrics_tbl)

write_csv(coef_tbl, "output/tables/multinom_coefs.csv")