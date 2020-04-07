library(magrittr)
library(lubridate)
library(stringr)
library(foreach)
library(data.table)
library(ggplot2)

# library(matlib)
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

# Set download directory
download.dir <- "~/Documents/GitHub/COVID-Experiments/CSV"
covid.url.latest <- "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"

# Load Download Directory Info
latest.file <- list.files(download.dir) %>% 
  str_subset("^\\d{4}-\\d{2}-\\d{2}\\.csv") %>% 
  sort %>% last

# Download CSV (if hasn't already been done today)
today.str <- today() %>% format(format="%Y-%m-%d")
if( latest.file != paste0(today.str,".csv") ) {
  download.file(covid.url.latest, today.str %>% paste0(".csv") %>% file.path(download.dir,.))
  latest.file <- paste0(today.str,".csv")
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
  
  covid.tbl[geoid %in% c("RU","TH","SG","QA","JP","AE"),is.train:=F]
  
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
  
  fwrite(covid.tbl,file.path(download.dir,str_glue("enriched-{today.str}.csv")))
  
  covid.train <- covid.tbl[(is.train)]
}


{
  ggplot(covid.tbl[is.train==T]) +
    geom_point(aes(x=date,y=log.cases)) +
    geom_smooth(aes(x=date,y=log.cases), se=F, color="red") +
    facet_wrap(~geoid) +
    scale_x_date(date_breaks = "10 days") +
    theme(axis.text.x = element_text(angle = 90))
  
    
  ggplot(covid.tbl[is.train==T][geoid %in% c("CN","KR","JP","US","CA","IR")]) +
    geom_point(aes(x=date,y=log.cases)) +
    geom_point(aes(x=date,y=adj.log.cases),colour="red") +
    facet_wrap(~geoid) +
    scale_x_date(date_breaks = "10 days") +
    theme(axis.text.x = element_text(angle = 90))
  
}


# Generate Density Graphs from Posterior Simulations
{
  train.ctrys <- covid.tbl[(is.train),geoid %>% unique]
  
  param.dir <- "~/Documents/GitHub/COVID-Experiments/Post_Sims/"
  latest.file <- list.files(param.dir) %>% sort %>% last
  beta.tbl <- file.path(param.dir,latest.file) %>% fread
  beta.tbl[,`:=`( peak.int = -b1/(2*b2),
                  V1=NULL )]
  beta.tbl[,`:=`( root1.int =  peak.int - (0.5/b2)*sqrt( (b1^2) - 4*b0*b2),
                  root2.int =  peak.int + (0.5/b2)*sqrt(b1^2 - 4*b0*b2) )]
  
  beta.tbl <- 
    beta.tbl %>%
    merge(covid.train[,.( ctry.min.date.int = ctry.min.date.int %>% first, 
                        country = country %>% first),by=geoid],
          by="geoid",
          all.x=T)
  
  beta.tbl[, c("peak.int","root1.int","root2.int") := .SD + ctry.min.date.int, 
           .SDcols = c("peak.int","root1.int","root2.int")]
  beta.tbl[, c("peak.date","root1.date","root2.date") := .SD %>% lapply(as_date), 
           .SDcols = c("peak.int","root1.int","root2.int")]
  
  prob.lst <- seq(0,1,0.025)
  cdf.tbl <-
    beta.tbl[, .(
      prb = prob.lst,
      root2.int = quantile(root2.int, probs = prob.lst),
      peak.int = quantile(peak.int, probs = prob.lst)
    ), by = geoid]
  
  cdf.tbl[peak.int<20000]
  
  cdf.tbl <- 
    cdf.tbl %>% 
    merge(covid.tbl[,.( ctry.min.date.int = ctry.min.date.int %>% first),by=geoid],
          all.x=T)
  
  cdf.tbl[,c("root2.int","peak.int") := .SD + ctry.min.date.int, .SDcols=c("root2.int","peak.int")]
  cdf.tbl[,c("root2.date","peak.date") := .SD %>% lapply(as_date), .SDcols=c("root2.int","peak.int")]
  
  plt.tbl <- cdf.tbl[,.(geoid,prb,root2.int,peak.int)] %>% melt(id.vars=c("geoid","prb"))
  
  ggplot(plt.tbl) +
    geom_line(aes(y=prb,x=value,colour = variable)) +
    facet_wrap(~geoid)
  
  ggplot( beta.tbl[peak.date>ymd(20200201)][peak.date<ymd(20200801) ] ) +
    geom_density(aes(x=peak.date), fill="red", alpha=0.7 ) +
    facet_wrap(~geoid,scales="free_x") +
    ggtitle("Peak Date Density") +
    scale_x_date(date_breaks = "15 days") +
    theme(axis.text.x = element_text(angle = 90))
  
  
  ca.tbl <- beta.tbl[peak.date>today()][peak.date<ymd(20200901) ][geoid %in% c("CA")]
  # ca.tbl <- beta.tbl[peak.date<ymd(20201001) ][geoid=="CA"]
  # ca.tbl <- beta.tbl[geoid=="CA"][,peak.date := min(peak.date,ymd(20200201))]
  
  ca.tbl.pctile <- quantile(ca.tbl$peak.int,seq(0.1,0.9,0.1)) %>% as_date
  ca.tbl.pctile2 <- quantile(ca.tbl$peak.int,c(0.5)) %>% as_date
  
  
  ggplot(ca.tbl) +
    geom_density(aes(x=peak.date), fill="blue", size = 0, alpha=0.5 ) +
    geom_vline(aes(xintercept=ca.tbl[,mean(peak.date)], colour = "Mean"), size=1.5) +
    geom_vline(aes(colour="Median", xintercept=ca.tbl.pctile2),  size=1.5) +
    geom_vline(aes(colour="10% Percentile Increments"), xintercept=ca.tbl.pctile, linetype="dotted", alpha=0.8, size=0.8) +
    ggtitle("Peak Date - Probability Density") +
    scale_x_date(date_breaks = "10 days") +
    theme(axis.text.x = element_text(angle = 90)) +
    facet_wrap(~country)
  
  
  
  ch.tbl <- beta.tbl[peak.date>ymd(20200320)][peak.date<ymd(20200601)][geoid == "CH"]

  ch.tbl.pctile <- quantile(ch.tbl$peak.int,seq(0.1,0.9,0.1)) %>% as_date
  ch.tbl.pctile2 <- quantile(ch.tbl$peak.int,c(0.5)) %>% as_date
  
  
  ggplot(ch.tbl) +
    geom_density(aes(x=peak.date), fill="blue", size = 0, alpha=0.5 ) +
    geom_vline(aes(xintercept=ch.tbl[,mean(peak.date)], colour = "Mean"), size=1.5) +
    geom_vline(aes(colour="Median", xintercept=ch.tbl.pctile2),  size=1.5) +
    geom_vline(aes(colour="10% Percentile Increments"), xintercept=ch.tbl.pctile, linetype="dotted", alpha=0.8, size=0.8) +
    ggtitle("Peak Date - Probability Density") +
    scale_x_date(date_breaks = "10 days") +
    theme(axis.text.x = element_text(angle = 90)) +
    facet_wrap(~country)
  

  
  ggplot(beta.tbl[peak.date<ymd(20201001)][geoid %in% c("CA","CH","US")]) +
    geom_density(aes(x=peak.date), fill="blue", size = 0, alpha=0.5 ) +
    geom_vline(aes(xintercept=mean(peak.date), colour = "Mean",group=geoid), size=1.5) +
    geom_vline(aes(colour="Median", xintercept=median(peak.date),group=geoid),  size=1.5) +
    ggtitle("Peak Date - Probability Density") +
    scale_x_date(date_breaks = "10 days") +
    theme(axis.text.x = element_text(angle = 90)) +
    facet_wrap(~country)
  
    
}


{
  
  # plot posterior predictions
  beta.mean.tbl <- beta.tbl[,.SD %>% lapply(mean),by=geoid,.SDcols=c("b0","b1","b2")]
  covid.train <- covid.train %>% merge(beta.mean.tbl,by = "geoid", all.x = T)
  covid.train[,log.cases.pred := b0 + (b1*date.int.delta) + (b2*(date.int.delta^2)) ]
  
  
  ggplot(covid.train) +
    geom_point(aes(x=date,y=adj.log.cases)) +
    geom_line(aes(x=date,y=log.cases.pred), color="red") +
    facet_wrap(~geoid) +
    scale_x_date(date_breaks = "5 days") +
    theme(axis.text.x = element_text(angle = 90))
  
  
  
}

