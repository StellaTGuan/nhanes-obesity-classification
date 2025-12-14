required_pkgs <- c(
  "nhanesA", "tidyverse", "nnet", "caret", "yardstick"
)

installed <- rownames(installed.packages())
to_install <- setdiff(required_pkgs, installed)
if (length(to_install) > 0) install.packages(to_install)

library(nhanesA)
library(tidyverse)
library(nnet)
library(caret)
library(yardstick)

set.seed(1)

demo <- nhanes("DEMO_L")
bmx  <- nhanes("BMX_L")

dat = merge(demo, bmx, by = "SEQN")

glimpse(dat)

dat_clean = dat[, c(
  "SEQN",
  "RIAGENDR",
  "RIDAGEYR",
  "DMDEDUC2",
  "DMDMARTZ",
  "RIDEXPRG",
  "INDFMPIR",
  "BMXBMI"
)]
glimpse(dat_clean)


