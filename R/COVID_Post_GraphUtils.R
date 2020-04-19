# Approximate data mode
getmode <- function(x) {
  d <- density(x)
  d$x[which.max(d$y)]
}

#' Generate aggregate graph of COVID training data
#' @param data.table of enriched covid data
#' @return ggplot object of training data
gen.train.data.graph <- function(x) {
  ggplot(x[is.train==T]) +
    geom_point(aes(x=date,y=log.cases)) +
    geom_smooth(aes(x=date,y=log.cases), se=F, color="red") +
    facet_wrap(~interaction(country,geoid)) +
    scale_x_date(date_breaks = "10 days") +
    theme(axis.text.x = element_text(angle = 90))
}


#' Generate Density Graphs from Posterior Simulations
#'
#' @return
#' @export Graphs as PDFs to parameter Directory of Choice
postPlotting.main <- function() {

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
  
  
  
  
  
# Canada Plots
  
  plt.dt <- beta.tbl #[peak.date>ymd(20200329)][peak.date<ymd(20200901) ]
  plt.dt <- plt.dt[,.(geoid,country,peak.int,root1.int)]
  plt.dt <- plt.dt %>% melt(measure.vars = c("peak.int","root1.int"), value.name = "value.int")
  plt.dt[, variable := variable %>% str_remove("\\.int")]
  plt.dt[, value.date := value.int %>% as_date]
  
  ggplot(plt.dt[geoid %in% c("CA","US","CH")]) +
    geom_density(aes(x=value.date), fill="purple", alpha=0.5, colour = 'black', size = 0.25 ) +
    geom_vline(aes(xintercept=mean(value.date), colour = "Mean"), size=1.5) +
    geom_vline(aes(colour="Median", xintercept=median(value.date)),  size=1.5) +
    geom_vline(aes(colour="Mode", xintercept=value.int %>% getmode %>% as_date ),  size=1.5) +
    ggtitle("Peak Date and End Date - Probability Density") +
    scale_x_date(date_breaks = "5 days") +
    theme(axis.text.x = element_text(angle = 90)) +
    facet_grid(variable~country)
  
  ggplot(ca.tbl) +
    geom_density(aes(x=root1.date), fill="purple", alpha=0.5, colour = 'black', size = 0.25 ) +
    geom_vline(aes(xintercept=ca.tbl[,mean(root1.date)], colour = "Mean"), size=1.5) +
    geom_vline(aes(colour="Median", xintercept=median(root1.date)),  size=1.5) +
    geom_vline(aes(colour="Mode", xintercept= root1.int %>% getmode %>% as_date ),  size=1.5) +
    ggtitle("Peak Date - Probability Density") +
    scale_x_date(date_breaks = "5 days") +
    theme(axis.text.x = element_text(angle = 90)) +
    facet_wrap(~country)
  
  
  ggplot(ca.tbl) +
    geom_density(aes(x=root1.date), fill="blue", size = 0, alpha=0.5 ) +
    geom_vline(aes(xintercept=mean(root1.date), colour = "Mean"), size=1.5) +
    geom_vline(aes(colour="Median", xintercept=median(root1.date)),  size=1.5) +
    ggtitle("Last Case Date Date - Probability Density") +
    scale_x_date(date_breaks = "5 days") +
    theme(axis.text.x = element_text(angle = 90)) +
    facet_wrap(~country)
  
  
  
  # plot posterior predictions
  beta.mean.tbl <- beta.tbl[,.SD %>% lapply(mean),by=geoid,.SDcols=c("b0","b1","b2")]
  covid.train <- covid.train %>% merge(beta.mean.tbl,by = "geoid", all.x = T)
  covid.train[,log.cases.pred := b0 + (b1*date.int.delta) + (b2*(date.int.delta^2)) ]
  
  
  ggplot(covid.train[geoid=="CA"]) +
    geom_point(aes(x=date,y=adj.log.cases)) +
    geom_line(aes(x=date,y=log.cases.pred), color="red") +
    facet_wrap(~geoid) +
    scale_x_date(date_breaks = "5 days") +
    theme(axis.text.x = element_text(angle = 90))
  
  
  
  
  
  
  
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
  
  
  
  ggplot(beta.tbl[peak.date<ymd(20201001)][peak.date>(today()-5)][geoid %in% c("CA","US","UK")]) +
    geom_density(aes(x=peak.date), fill="blue", size = 0, alpha=0.5 ) +
    # geom_vline(aes(xintercept=mean(peak.date), colour = "Mean",group=interaction(geoid)), size=1.5) +
    # geom_vline(aes(colour="Median", xintercept=median(peak.date),group=interaction(geoid)),  size=1.5) +
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



