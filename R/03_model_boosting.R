# ============================================================
# 03_model_boosting.R
# Gradient Boosting Trees (GBM) for 3-class BMI classification
# ASSUMES cleaned dataset has NO NAs (complete cases)
# ============================================================

required_pkgs <- c("tidyverse", "readr", "caret", "gbm", "yardstick")
installed <- rownames(installed.packages())
to_install <- setdiff(required_pkgs, installed)
if (length(to_install) > 0) install.packages(to_install)

library(tidyverse)
library(readr)
library(caret)
library(gbm)
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
candidate_factor <- c("sex", "educ", "marital")
candidate_num <- c(
  "age", "pir", "alcohol_drinks_per_day", "sleep_hours_avg", "mvpa_eq_min_wk",
  "kcal_day1", "fiber_day1", "satfat_day1"
)

factor_cols <- intersect(candidate_factor, names(dat))
num_cols    <- intersect(candidate_num, names(dat))

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
# GBM model (caret)
# -------------------------
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE)

grid <- expand.grid(
  n.trees = c(100, 300, 600, 1000),
  interaction.depth = c(1, 2, 3),
  shrinkage = c(0.01, 0.05),
  n.minobsinnode = c(10, 20)
)

fit_gbm <- train(
  bmi_cat3 ~ .,
  data = train_mod,
  method = "gbm",
  trControl = ctrl,
  tuneGrid = grid,
  metric = "Accuracy",
  verbose = FALSE
)

write_csv(fit_gbm$results, "output/tables/gbm_cv_results.csv")
write_csv(as_tibble(fit_gbm$bestTune), "output/tables/gbm_best_tune.csv")

# -------------------------
# Test evaluation
# -------------------------
pred_class <- predict(fit_gbm, newdata = test_mod)

cm <- caret::confusionMatrix(pred_class, test_mod$bmi_cat3)
sink("output/tables/gbm_confusion_matrix.txt")
print(cm)
sink()

test_eval <- tibble(truth = test_mod$bmi_cat3, estimate = pred_class)
metrics_tbl <- tibble(
  accuracy = yardstick::accuracy_vec(test_eval$truth, test_eval$estimate),
  macro_f1  = yardstick::f_meas_vec(test_eval$truth, test_eval$estimate, estimator = "macro")
)
write_csv(metrics_tbl, "output/tables/gbm_test_metrics.csv")

cm_df <- as.data.frame(cm$table)
p_cm <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 4) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix (GBM)", x = "True class", y = "Predicted class") +
  theme_minimal(base_size = 13) +
  theme(panel.grid = element_blank())

ggsave("output/figures/gbm_confusion_matrix.png", p_cm, width = 6.5, height = 5, dpi = 300)

# -------------------------
# CV curve figure (error vs #trees) for best combo
# -------------------------
cv <- fit_gbm$results
best <- fit_gbm$bestTune

cv_plot_df <- cv %>%
  filter(
    interaction.depth == best$interaction.depth,
    shrinkage == best$shrinkage,
    n.minobsinnode == best$n.minobsinnode
  ) %>%
  mutate(cv_error = 1 - Accuracy) %>%
  arrange(n.trees)

p_cv <- ggplot(cv_plot_df, aes(x = n.trees, y = cv_error)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(
    title = "GBM Cross-Validated Error vs Number of Trees (Best Hyperparameters)",
    x = "Number of trees (n.trees)",
    y = "Cross-validated error (1 - Accuracy)"
  ) +
  theme_minimal(base_size = 13) +
  theme(panel.grid.minor = element_blank())

ggsave("output/figures/gbm_cv_curve.png", p_cv, width = 7.5, height = 5, dpi = 300)

# -------------------------
# Variable importance figure
# -------------------------
vi <- varImp(fit_gbm, scale = FALSE)$importance %>%
  rownames_to_column("feature") %>%
  as_tibble() %>%
  arrange(desc(Overall))

write_csv(vi, "output/tables/gbm_var_importance.csv")

top_k <- 15
p_vi <- vi %>%
  slice_head(n = top_k) %>%
  ggplot(aes(x = reorder(feature, Overall), y = Overall)) +
  geom_col() +
  coord_flip() +
  labs(title = paste0("GBM Variable Importance"),
       x = NULL, y = "Importance") +
  theme_minimal(base_size = 13)

ggsave("output/figures/gbm_var_importance.png", p_vi, width = 7.0, height = 5.5, dpi = 300)

cat("\n=== GBM (complete cases) ===\n")
print(fit_gbm)
cat("\nTest metrics:\n")
print(metrics_tbl)
cat("\nBest tuning parameters:\n")
print(fit_gbm$bestTune)