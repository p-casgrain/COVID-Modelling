# Set project paths
PROJECT_ROOT <- "~/Documents/GitHub/COVID-Experiments" # ATTENTION: You need to change this to the location of the project root 
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
  
  # == Generate COVID data
  covid <- generate.covid.data(PROJECT_CSV_DIR)
  covid.train <- covid[is.train==T]
  
  # == Run Python model training script
  py.script.path <- file.path(PROJECT_PY_DIR,"COVID_ModelFit_LM.py")
  py_run_string("import os; os.chdir( os.path.expanduser('{PROJECT_ROOT}') )" %>% str_glue)
  py_run_string("print( 'Current python working directory is %s' % os.getcwd() )")
  py_run_string("n_iter = int(1e4); print_interval = int(5e3); n_samples = int(3e4);")
  source_python(py.script.path, envir = parent.frame(), convert = FALSE)

  # == Load Posterior Simulations Data
  beta.tbl <- load.latest.beta.sims(covid.train %>% copy) %>% copy
  beta.tbl <- beta.tbl[ (maxroot.date<ymd(20201001)) &
                          (peak.date<ymd(20201001)) ]
  
  # == Generate density and posterior prediction plots
  
  # Set countries to plot
  geoid.lst <- c("CA","US","CH","IT")
  
  # Generate Plots
  gen.density.plots( beta.tbl[geoid %in% geoid.lst])
  
  
  gen.posterior.preds( beta.tbl[geoid %in% geoid.lst], 
                       covid.train[geoid %in% geoid.lst] )
  

}

