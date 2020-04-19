# Set project paths
PROJECT_ROOT <- "~/Documents/GitHub/COVID-Experiments"
PROJECT_R_DIR <- file.path(PROJECT_ROOT,"R")
PROJECT_CSV_DIR <- file.path(PROJECT_ROOT,"CSV")
PROJECT_PY_DIR <- file.path(PROJECT_ROOT,"Python")

# Import and source libraries
library(reticulate)
use_python("/usr/local/anaconda3/bin/python",required=T)

setwd(PROJECT_ROOT)
source(file.path(PROJECT_R_DIR,"COVID_DataGrab.R"))
source(file.path(PROJECT_R_DIR,"COVID_Post_GraphUtils.R"))


main <- function() {
  
  # Generate new COVID data
  covid <- generate.covid.data(PROJECT_CSV_DIR)
  covid.train <- covid[is.train==T]
  
  # Run Python model training script (very sketchy)
  py.script.path <- file.path(PROJECT_PY_DIR,"COVID_ModelFit_LM.py")
  py_run_string("n_iter = int(2e5) \nprint_interval = 50")
  source_python(py.script.path, envir = parent.frame(), convert = FALSE)
  
  # Plot some graphs (Focus on Canada, Switzerland and USA)
  gen.train.data.graph(covid.train[geoid %in% c("CA","US","CH")])
  
  
  
}