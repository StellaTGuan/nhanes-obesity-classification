# 01_clean_nhanes.R

### Variables and Files
## Merge Key
"SEQN"

## Outcome
# BMX_L.xpt: outcome, BMI data

## Demographical
# DEMO_L.xpt 
# "RIAGENDR", "RIDAGEYR", "DMDEDUC2", "DMDMARTZ", "RIDEXPRG", "INDFMPIR",


## Lifestyle Variables
# PAQ_L.xpt: Physical Exercises
  # only EXCLUDE PAD680
# ALQ_L.xpt: 
  # 0 if ALQ111 == No (never drank) or ALQ121 indicates no drinking in past 12 months
  # else ALQ130
# SLQ_L.xpt: Sleep
  # (5*SLD012 + 2*SLD013)/7
  # SLD012 = sleep hours on weekdays/workdays
  # SLD013 = sleep hours on weekends

required_pkgs <- c(
  "nhanesA", "tidyverse", "nnet", "caret", "yardstick"
)

# only run once if not installed
installed <- rownames(installed.packages())
to_install <- setdiff(required_pkgs, installed)
if (length(to_install) > 0) install.packages(to_install)

library(nhanesA)
library(tidyverse)
library(nnet)
library(caret)
library(yardstick)
library(dplyr)
library(stringr)
library(readr)
library(ggplot2)
library(scales)


## ----------------------------------------------
## Load needed components & Keep only needed cols
## ----------------------------------------------

## Demographics
demo <- nhanes("DEMO_L") %>%
  select(SEQN, RIAGENDR, RIDAGEYR, DMDEDUC2, DMDMARTZ, RIDEXPRG, INDFMPIR)

## Outcome (BMI)
bmx <- nhanes("BMX_L") %>%
  select(SEQN, BMXBMI)

## Smoking: SMQ040
smq <- nhanes("SMQ_L") %>%
  select(SEQN, SMQ040)

## Physical exercise: keep these, EXCLUDE PAD680
paq <- nhanes("PAQ_L") %>%
  select(SEQN, PAD790Q, PAD790U, PAD800, PAD810Q, PAD810U, PAD820)  # excludes PAD680

## Alcohol
alq <- nhanes("ALQ_L") %>%
  select(SEQN, ALQ111, ALQ121, ALQ130)

## Sleep
slq <- nhanes("SLQ_L") %>%
  select(SEQN, SLD012, SLD013)

## -----------------------
## Merge (SEQN key)
## -----------------------
dat_rawvars <- demo %>%
  inner_join(bmx, by = "SEQN") %>% # keep BMI
  left_join(smq, by = "SEQN") %>%
  left_join(paq, by = "SEQN") %>%
  left_join(alq, by = "SEQN") %>%
  left_join(slq, by = "SEQN")

glimpse(dat_rawvars)




# --- helper: NHANES special missing codes -> NA (only matters if they exist in your numeric cols)
nhanes_to_na <- function(x) {
  if (is.numeric(x)) {
    x[x %in% c(".", 7, 9, 77, 99, 777, 999, 7777, 9999)] <- NA
  }
  x
}


# Convert unit text -> multiplier to get "times per week"
unit_to_week <- function(u_chr) {
  case_when(
    str_to_lower(u_chr) == "day"   ~ 7,
    str_to_lower(u_chr) == "week"  ~ 1,
    str_to_lower(u_chr) == "month" ~ 1/4.345,
    str_to_lower(u_chr) == "year"  ~ 1/52,
    TRUE ~ NA_real_
  )
}

dat2 <- dat_rawvars %>%
  # ----------------------------
# 1) Rename columns (cleaner)
# ----------------------------
rename(
  id              = SEQN,
  sex             = RIAGENDR,
  age             = RIDAGEYR,
  educ            = DMDEDUC2,
  marital         = DMDMARTZ,
  pregnant_status = RIDEXPRG,
  pir             = INDFMPIR,
  bmi             = BMXBMI,
  
  mod_freq        = PAD790Q,
  mod_unit        = PAD790U,
  mod_mins        = PAD800,
  vig_freq        = PAD810Q,
  vig_unit        = PAD810U,
  vig_mins        = PAD820,
  
  alc_ever        = ALQ111,
  alc_freq12m     = ALQ121,
  alc_drinks_day  = ALQ130,
  
  sleep_wkday     = SLD012,
  sleep_wkend     = SLD013
) %>%
  # ---------------------------------------------------------
# 2) Standardize obvious "non-information" to NA
#    - educ/marital: "Don't know" / "Refused" -> NA
#    - activity units: blank string "" -> NA
# ---------------------------------------------------------
mutate(
  educ    = na_if(educ, "Don't know"),
  marital = na_if(marital, "Don't know"),
  marital = na_if(marital, "Refused"),
  
  mod_unit = na_if(as.character(mod_unit), ""),
  vig_unit = na_if(as.character(vig_unit), "")
) %>%
  # ---------------------------------------------------------
# 3) Numeric conversion + special-missing handling
#    - ALQ130 has 777 / 999 as special missing
# ---------------------------------------------------------
mutate(
  across(c(age, pir, bmi, mod_freq, mod_mins, vig_freq, vig_mins,
           alc_drinks_day, sleep_wkday, sleep_wkend),
         ~ as.numeric(.)),
  
  alc_drinks_day = if_else(alc_drinks_day %in% c(777, 999), NA_real_, alc_drinks_day)
) %>%
  # ---------------------------------------------------------
# 4) Sanity checks (set out-of-range to NA, do NOT drop rows)
#    - age: [0, 120]
#    - bmi: [10, 80]  (broad but reasonable for adult BMI)
#    - pir: [0, 5]    (NHANES PIR is typically capped at 5)
#    - sleep hours: [0, 24]
#    - mins/session: [0, 720]  (0–12 hours per session; very generous)
#    - freq: [0, 365]          (max "per day" frequency across a year)
#    - drinks/day: [0, 50]     (very generous upper bound)
# ---------------------------------------------------------
mutate(
  age  = if_else(age  >= 0  & age  <= 120, age, NA_real_),
  bmi  = if_else(bmi  >= 10 & bmi  <= 80,  bmi, NA_real_),
  pir  = if_else(pir  >= 0  & pir  <= 5,   pir, NA_real_),
  
  sleep_wkday = if_else(sleep_wkday >= 0 & sleep_wkday <= 24, sleep_wkday, NA_real_),
  sleep_wkend = if_else(sleep_wkend >= 0 & sleep_wkend <= 24, sleep_wkend, NA_real_),
  
  mod_mins = if_else(mod_mins >= 0 & mod_mins <= 720, mod_mins, NA_real_),
  vig_mins = if_else(vig_mins >= 0 & vig_mins <= 720, vig_mins, NA_real_),
  
  mod_freq = if_else(mod_freq >= 0 & mod_freq <= 365, mod_freq, NA_real_),
  vig_freq = if_else(vig_freq >= 0 & vig_freq <= 365, vig_freq, NA_real_),
  
  alc_drinks_day = if_else(alc_drinks_day >= 0 & alc_drinks_day <= 50, alc_drinks_day, NA_real_)
) %>%
  # ---------------------------------------------------------
# 5) Feature engineering (handles your exact distinct values)
# ---------------------------------------------------------
mutate(
  # Pregnancy flag (you said you'll later exclude pregnant participants)
  pregnant_at_exam = case_when(
    pregnant_status == "Yes, positive lab pregnancy test or self-reported pregnant at exam" ~ 1,
    pregnant_status == "The participant was not pregnant at exam"                           ~ 0,
    pregnant_status == "Cannot ascertain if the participant is pregnant at exam"            ~ NA_real_,
    TRUE                                                                                   ~ NA_real_
  ),
  
  # Alcohol: single feature "alcohol_drinks_per_day"
  # 0 if never drank OR never in last year; else use ALQ130 (drinks/day)
  alcohol_drinks_per_day = case_when(
    alc_ever == "No"                          ~ 0,
    alc_freq12m == "Never in the last year"   ~ 0,
    alc_ever == "Yes"                         ~ alc_drinks_day,
    TRUE                                      ~ NA_real_
  ),
  
  # Sleep: weighted avg; if one of weekday/weekend missing, use the available one
  sleep_hours_avg = {
    w_wkday <- if_else(!is.na(sleep_wkday), 5, 0)
    w_wkend <- if_else(!is.na(sleep_wkend), 2, 0)
    denom <- w_wkday + w_wkend
    if_else(denom > 0,
            (w_wkday * sleep_wkday + w_wkend * sleep_wkend) / denom,
            NA_real_)
  },
  
  # Physical activity: MVPA-equivalent minutes/week
  mod_sessions_wk = case_when(
    mod_freq == 0 ~ 0,
    mod_freq > 0  ~ mod_freq * unit_to_week(mod_unit),
    TRUE          ~ NA_real_
  ),
  vig_sessions_wk = case_when(
    vig_freq == 0 ~ 0,
    vig_freq > 0  ~ vig_freq * unit_to_week(vig_unit),
    TRUE          ~ NA_real_
  ),
  
  mod_min_wk = case_when(
    mod_freq == 0 ~ 0,
    mod_freq > 0 & !is.na(mod_sessions_wk) & !is.na(mod_mins) ~ mod_sessions_wk * mod_mins,
    TRUE ~ NA_real_
  ),
  vig_min_wk = case_when(
    vig_freq == 0 ~ 0,
    vig_freq > 0 & !is.na(vig_sessions_wk) & !is.na(vig_mins) ~ vig_sessions_wk * vig_mins,
    TRUE ~ NA_real_
  ),
  
  mvpa_eq_min_wk = case_when(
    is.na(mod_min_wk) & is.na(vig_min_wk) ~ NA_real_,
    TRUE ~ coalesce(mod_min_wk, 0) + 2 * coalesce(vig_min_wk, 0)
  )
) %>%
  # Keep modeling columns (+ raw BMI)
  select(
    id, sex, age, educ, marital, pir, bmi,
    pregnant_at_exam,
    alcohol_drinks_per_day,
    sleep_hours_avg,
    mvpa_eq_min_wk
  )


dat2 <- dat2 %>%
  filter(
    !is.na(bmi),                                # make sure no NA in outcome var 
    age >= 20,                                  # only adults 20+
    is.na(pregnant_at_exam) | pregnant_at_exam == 0  # exclude confirmed pregnant
  )%>%
  select(-pregnant_at_exam)

# Quick check
glimpse(dat2)

# --- Save cleaned, feature-ready dataset ---
write_csv(
  dat2,
  file = "data/processed/nhanes_adults20_bmi_features.csv",
  na = ""
)


# Create BMI category (adult cutoffs) and count distribution
bmi_dist <- dat2 %>%
  filter(!is.na(bmi)) %>%
  mutate(
    bmi_cat = case_when(
      bmi < 25   ~ "Underweight & Normal",
      bmi < 30   ~ "Overweight",
      TRUE       ~ "Obese"
    ),
    bmi_cat = factor(bmi_cat, levels = c("Underweight & Normal", "Overweight", "Obese"))
  ) %>%
  count(bmi_cat) %>%
  mutate(pct = n / sum(n))

p <- ggplot(bmi_dist, aes(x = bmi_cat, y = n)) +
  geom_col(width = 0.35, fill = "lightblue") +  # narrower bars + blue
  geom_text(aes(label = percent(pct, accuracy = 0.1)),
            vjust = 1.4, color = "black", size = 4.2) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.05))) +
  labs(
    title = "BMI Category Distribution (Adults 20+, Non-Pregnant)",
    x = NULL,
    y = "Number of Participants"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid = element_blank(),          # remove ALL gridlines
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(size = 11)
  )

# Save figure
ggsave(
  filename = "output/figures/bmi_category_distribution.png",
  plot = p,
  width = 7,
  height = 5,
  dpi = 300
)

