library(magrittr)
library(lubridate)
library(stringr)
library(foreach)
library(data.table)
library(ggplot2)

library(matlib)
library(glmnet)
library(glmnetUtils)
library(mvtnorm)


# Define utility functions
rename <- function(x,new.names){
  names(x) = new.names
  return(x)
}

`%!in%` <- function(x,y) !(x %in% y)

# Set download directory
download.dir <- "~/Documents/GitHub/COVID-Experiments/CSV"
covid.url.latest <- "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"

# Load Download Directory Info
latest.file <- list.files(download.dir) %>% 
  str_subset("\\d{4}-\\d{2}-\\d{2}\\.csv") %>% 
  sort %>% first

# Download CSV (if hasn't already been done today)
today.str <- today() %>% format(format="%Y-%m-%d")
if( latest.file != paste0(today.str,".csv") ) {
  download.file(covid.url.latest, today.str %>% paste0(".csv") %>% file.path(download.dir,.))
}

# Load Latest CSV
file.path <- file.path(download.dir,latest.file)
covid.tbl <- fread(file.path)

# Format & Enrich Table (may need to change since ECDC changes format all of the time)
{
  setnames( covid.tbl,
            c("dateRep","popData2018","countriesAndTerritories"),
            c("date.str","population","country"))
  
  setnames( covid.tbl,
            covid.tbl %>% colnames,
            covid.tbl %>% colnames %>% tolower)
  
  covid.tbl[,`:=`( date = dmy(date.str),
                   date.int = dmy(date.str) %>% as.integer,
                   country = country %>% tolower,
                   log.cases = log(cases),
                   log.deaths = log(deaths),
                   date.str = NULL, day = NULL, 
                   month = NULL, year = NULL)]
  
  covid.tbl <- covid.tbl[order(country,date)]
  
  covid.tbl[, `:=`( tot.cases = sum(cases),
                    tot.deaths = sum(deaths),
                    cum.cases = cumsum(cases),
                    cum.deaths = cumsum(deaths),
                    ctry.min.date.int = min(date.int),
                    country = tolower(country),
                    n.nonzero.rows = (0==cases) %>% not %>% sum,
                    prop.zero.cases = mean(0==cases),
                    prop.zero.deaths = mean(0==deaths) ) , 
            by = .(geoid)]
  
  covid.tbl[cases>0, delta.next.non0 := c( diff(date.int), 1 ), by=geoid]
  covid.tbl[cases>0, max.gap := delta.next.non0 %>% rev %>% cummax %>% rev, by=geoid]
  
  fwrite(covid.tbl,file.path(download.dir,str_glue("enriched-{today.str}.csv")))
}

# Generate Training Data Set
{
  covid.tbl.train <- covid.tbl[(cases>0)&(!is.na(cases))&
                                 (tot.cases>1000)&(n.nonzero.rows>20)&
                                 (max.gap<4)&(geoid %!in% c("RU"))]
  
  covid.tbl.train[, log.cases := log(cases) ]
  
  train.ctrys <- covid.tbl.train[,geoid %>% unique]
  
}


# Fit Linear Models
{
  # Define Formula
  model.fmla <- log.cases ~ 1 + I(date.int-ctry.min.date.int) + I((date.int-ctry.min.date.int)^2)
  
  # Fit Constrained Model per Country
  model.lst <- 
    foreach(id=train.ctrys) %do%
    { glmnet(formula = model.fmla,
             data = covid.tbl.train[geoid==id],
             upper.limits	= c(Inf,Inf,0),
             use.model.frame=T,
             lambda = 0) }
  
  model.lst %<>% rename(train.ctrys)
  
  # Generate Model Predictions
  covid.tbl.train[, pred.log.cases := predict( model.lst[[first(geoid)]], .SD ),
                  by = geoid]
  
  # Store Coefficients
  model.coefs.tbl <- model.lst %>% sapply(function(x) x %>% coef %>% as.matrix ) %>% t %>% data.table(keep.rownames = T)
  setnames(model.coefs.tbl,colnames(model.coefs.tbl),c("geoid","b0","b1","b2"))
  
  # Plot results
  ggplot(covid.tbl.train) + 
    geom_point(aes(x=date,y=cases %>% log )) +
    geom_line(aes(x=date,y=pred.log.cases), colour='red') +
    facet_wrap(~country,scales="free")
}


# Fit Gaussian RVs to Each Coefficient
coef.covmat <- model.coefs.tbl[,cbind(b0,b1,b2) %>% cov]
coef.covmat.inv <- coef.covmat %>% inve
coef.mean <- model.coefs.tbl[,.(b0,b1,b2) %>% sapply(mean)]



get.quad.max <- ( function(cfs) -cfs["1"]/(2*cfs["2"]) ) %>% Vectorize
get.quad.roots <- ( function(cfs) ( -cfs["1"] + c(-1,1)*sqrt(cfs["1"]^2 - 4*cfs["0"]*cfs["2"]) )/(2*cfs["2"]) ) %>% Vectorize
get.quad.root.gap <- ( function(cfs) ( sqrt(cfs["1"]^2 - 4*cfs["0"]*cfs["2"]) )/cfs["2"] ) %>% Vectorize


model.coefs %>% get.quad.max
model.coefs %>% get.quad.roots
model.coefs %>% get.quad.root.gap





ggplot(covid.tbl[geoid=="CN"]) +
  geom_point(aes(x=date,y=cases,colour=geoid))


