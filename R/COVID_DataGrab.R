library(magrittr)
library(lubridate)
library(stringr)
library(foreach)
library(data.table)
library(ggplot2)

library(glmnet)
library(glmnetUtils)
library(mvtnorm)
library(scales)



# Define utility functions
rename <- function(x,new.names){
  names(x) = new.names
  return(x)
}
`%!in%` <- function(x,y) !(x %in% y)


#' ECDC COVID Data Grab and Enrich
#' Download COVID Data from ECDC, Format data for model training and write enriched data to disk as CSVs
#'
#' @param covid.csv.dir directory in which to download and write csvs for covid data
#'
#' @return data.table of enriched covid data
#' @export writes CSVs to download.dir and subdirectories
#'
#' @examples generate.covid.data("~/Documents/GitHub/COVID-Experiments/CSV")
generate.covid.data <- function(covid.csv.dir = getwd() ){
    
    # Hard coded covid data URL
    covid.url.latest <- "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
    
    # Get latest covid data as a data.table
    {
      # Load Download Directory Info
      latest.file <- list.files(covid.csv.dir) %>% 
        str_subset("^\\d{4}-\\d{2}-\\d{2}\\.csv") %>% 
        sort %>% last
      
      # Download CSV (if hasn't already been done today)
      today.str <- today() %>% format(format="%Y-%m-%d")
      if( latest.file != paste0(today.str,".csv") ) {
        str_glue("Downloading new data for date {today.str} to {covid.csv.dir}. \n\n") %>% cat
        download.file(covid.url.latest, today.str %>% paste0(".csv") %>% file.path(covid.csv.dir,.))
        latest.file <- paste0(today.str,".csv")
      }
      
      # Load Latest CSV
      file.path <- file.path(covid.csv.dir,latest.file)
      covid.tbl <- fread(file.path)
      str_glue("Loaded COVID Data from path {file.path}. \n\n") %>% cat
    }
    
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
      
      covid.tbl <- covid.tbl[order(geoid,date)]
      
      covid.tbl[, `:=`( tot.cases = sum(cases),
                        tot.deaths = sum(deaths),
                        cum.cases = cumsum(cases),
                        cum.deaths = cumsum(deaths),
                        country = tolower(country),
                        n.nonzero.rows = (0==cases) %>% not %>% sum,
                        prop.zero.cases = mean(0==cases),
                        prop.zero.deaths = mean(0==deaths) ) , 
                by = geoid]
      
      covid.tbl[cases>0, delta.next.non0 := c( diff(date.int), 1 ), by=geoid]
      covid.tbl[cases>0, max.gap := delta.next.non0 %>% rev %>% cummax %>% rev, by=geoid]
    }
    
    # Generate Training Data Set and Further Enrichment
    {
      covid.tbl[,is.train := (cases>0)&(!is.na(cases))&
                  (tot.cases>1000)&(n.nonzero.rows>20)&(max.gap<4)]
      
      covid.tbl[geoid %in% c("RU","TH","SG","QA","JP","AE",
                             "KZ","DO","IQ","EC","BH","EE",
                             "DZ","MD","PE","BY","HR","HU",
                             "KW","ID","RS","AM","RS","SL",
                             "UA","OM","UZ","LT","MA","PR",
                             "BA","BD"),is.train:=F]
      
      covid.tbl[(geoid=="KR")&(date>ymd(20200314)),is.train:=F]
      covid.tbl[(geoid=="CN")&(date>ymd(20200316)),is.train:=F]
      covid.tbl[(geoid=="US")&(date<ymd(20200301)),is.train:=F]
      covid.tbl[(geoid=="EC")&(date<ymd(20200310)),is.train:=F]
      covid.tbl[(geoid=="PK")&(date<ymd(20200305)),is.train:=F]
      
      covid.tbl[(is.train),`:=`( ctry.min.date.int = min(date.int),
                                 ctry.min.date = min(date) ),by=geoid]
      
      covid.tbl[(is.train),`:=`( date.int.delta = (date.int-ctry.min.date.int),
                                 ctry.first.log.cases = log.cases - first(log.cases),
                                 adj.log.cases = log.cases - first(log.cases) ),by=geoid]
      
      
      
      # Write enriched covid data to disk
      enriched.write.path <- file.path(covid.csv.dir,str_glue("enriched-{today.str}.csv"))
      fwrite(covid.tbl,enriched.write.path)
      str_glue("Wrote enriched covid data to {enriched.write.path} . \n\n") %>% cat
      
    }
    
    return(covid.tbl)
  }


