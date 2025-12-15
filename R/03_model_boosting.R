# ============================================================
# 03_model_boosting.R
# Gradient Boosting Trees (GBM) for 3-class BMI classification
# Outputs: CV tuning table, test metrics, confusion matrix fig,
#          CV curve fig (error vs #trees), variable importance fig
# ============================================================

# -------------------------
# Packages
# -------------------------
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

# -------------------------
# Load data
# -------------------------
path <- "data/processed/nhanes_adults20_bmi_features.csv"
dat <- read_csv(path, show_col_types = FALSE)

# -------------------------
# Outcome: 3-class BMI (safe factor levels)
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
candidate_factor <- c("sex", "educ", "marital")
factor_cols <- intersect(candidate_factor, names(train))

candidate_num <- c(
  "age",
  "pir",          # income-related (poverty-income ratio)
  "alcohol_drinks_per_day",
  "sleep_hours_avg",
  "mvpa_eq_min_wk",
  # NEW dietary predictors
  "kcal_day1",
  "fiber_day1",
  "satfat_day1"
)
num_cols <- intersect(candidate_num, names(train))

make_unknown_factor <- function(x) {
  x <- as.character(x)
  x[is.na(x) | x == ""] <- "Unknown"
  factor(x)
}

if (length(factor_cols) > 0) {
  train <- train %>% mutate(across(all_of(factor_cols), make_unknown_factor))
  test  <- test  %>% mutate(across(all_of(factor_cols), make_unknown_factor))
}

if (length(num_cols) > 0) {
  train_medians <- sapply(train[num_cols], function(v) median(v, na.rm = TRUE))
  
  for (v in num_cols) {
    flag <- paste0(v, "_na")
    train[[flag]] <- is.na(train[[v]])
    test[[flag]]  <- is.na(test[[v]])
    
    train[[v]] <- if_else(is.na(train[[v]]), train_medians[[v]], train[[v]])
    test[[v]]  <- if_else(is.na(test[[v]]),  train_medians[[v]], test[[v]])
  }
}

# Keep only modeling columns (don’t use id or raw bmi)
keep_vars <- c("bmi_cat3", factor_cols, num_cols, paste0(num_cols, "_na"))
keep_vars <- keep_vars[keep_vars %in% names(train)]
train_mod <- train %>% select(all_of(keep_vars))
test_mod  <- test  %>% select(all_of(keep_vars))

# -------------------------
# GBM model (caret)
# Key tuning params match lecture:
# - n.trees (B), shrinkage (learning rate), interaction.depth (d)
# -------------------------
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE
)

# Keep grid moderate so it runs reasonably fast
grid <- expand.grid(
  n.trees = c(100, 300, 600, 1000),
  interaction.depth = c(1, 2, 3),
  shrinkage = c(0.01, 0.05),
  n.minobsinnode = c(10, 20)
)

set.seed(1)
fit_gbm <- train(
  bmi_cat3 ~ .,
  data = train_mod,
  method = "gbm",
  trControl = ctrl,
  tuneGrid = grid,
  metric = "Accuracy",
  verbose = FALSE
)

# Save CV results + chosen tuning parameters
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
  macro_f1 = yardstick::f_meas_vec(test_eval$truth, test_eval$estimate, estimator = "macro")
)
write_csv(metrics_tbl, "output/tables/gbm_test_metrics.csv")

# Confusion matrix figure
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
# CV curve figure (like lecture: error vs #trees, for best (depth, shrinkage, minobs))
# We'll plot CV error = 1 - Accuracy using only the best combo,
# so the plot is a single clean curve (not multiple groups/panels).
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
# Variable importance figure (report-friendly)
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
  labs(title = paste0("GBM Variable Importance (Top ", top_k, ")"),
       x = NULL, y = "Importance") +
  theme_minimal(base_size = 13)

ggsave("output/figures/gbm_var_importance.png", p_vi, width = 7.0, height = 5.5, dpi = 300)

# -------------------------
# Quick console output
# -------------------------
cat("\n=== GBM (Gradient Boosting Trees) ===\n")
print(fit_gbm)
cat("\nTest metrics:\n")
print(metrics_tbl)
cat("\nBest tuning parameters:\n")
print(fit_gbm$bestTune)
