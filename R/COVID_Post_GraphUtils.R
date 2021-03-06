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


#' Generate Posterior Density plots for COVID Model
#'
#' @param in.beta.tbl data.table of simulated posterior coefficients
#'
#' @return ggplot graph
gen.density.plots <- 
  function(in.beta.tbl) {
    
    plt.dt <- in.beta.tbl[,.(geoid,country,peak.int,maxroot.int)]
    plt.dt <- plt.dt %>% melt(measure.vars = c("peak.int","maxroot.int"), value.name = "value.int")
    plt.dt[,`:=`( variable = variable %>% str_remove("\\.int"),
                  value.date = value.int %>% as_date ) ]
    plt.dt[, `:=`(
      value.date.mean = value.int %>% mean %>% as_date,
      value.date.median = value.int %>% median %>% as_date,
      value.date.mode = value.int %>% getmode %>% as_date
    ),
    by = .(geoid, variable)]
    
    label.lst <- list(peak="Peak New Cases per Day",maxroot="Last New Case")
    for (nm in names(label.lst)) {
      plt.dt[variable==nm, varName := label.lst[nm] ]
    }
    
    ggplot(plt.dt) +
      geom_density(aes(x=value.date, group=interaction(variable,geoid)), 
                   fill="purple", alpha=0.4, colour = 'black', size = 0.25 ) +
      geom_vline(aes(colour="Mean", xintercept=value.date.mean ), size=0.75) +
      geom_vline(aes(colour="Median", xintercept=value.date.median ),  size=0.75) +
      geom_vline(aes(colour="Mode", xintercept=value.date.mode ),  size=0.75) +
      ggtitle("Peak Date and End Date - Probability Densities") +
      theme(axis.text.x = element_text(angle = 90)) +
      facet_grid(varName~geoid, scales="free_y") +
      # xlab("Date") +
      scale_x_date(name="Date", date_breaks = "15 days") +
      scale_y_continuous(name = "Empirical Density")
  }


#' Compute Various Predictive Statistics
#'
#' @param in.covid.tbl 
#' @param in.beta.tbl 
#'
#' @return data.table of stats
compute.predictions <- 
  function(in.covid.tbl,in.beta.tbl) {
  
  uq.geoid <- in.covid.tbl$geoid %>% unique
    
  # Get geoid subset and date deltas
  date.deltas <- in.covid.tbl[, date.int.delta %>% unique %>% sort]
  
  # Subset Beta Simulations
  in.beta.tbl <- in.beta.tbl[,.(geoid,country,b0,b1,b2)]
  
  # Create data table to store results 
  pred.stat.tbl <- CJ( geoid=uq.geoid, date.int.delta=date.deltas, sorted=F )
  
  # Loop through geoids
  for(gid in uq.geoid) {
    tmp.beta.mat <- in.beta.tbl[geoid==gid, .(b0,b1,b2) ] %>% as.matrix
    tmp.delta.mat <- pred.stat.tbl[geoid==gid,.(1,date.int.delta^1,date.int.delta^2)] %>% as.matrix
    tmp.pred.mat <- tmp.beta.mat %*% (tmp.delta.mat %>% t) 
    pred.stat.tbl[geoid==gid, mode := tmp.pred.mat %>% apply(2,getmode)]
    pred.stat.tbl[geoid==gid, median := tmp.pred.mat %>% apply(2,median)]
    pred.stat.tbl[geoid==gid, mean := tmp.pred.mat %>% apply(2,mean)]
    pred.stat.tbl[geoid==gid, qtile25 := tmp.pred.mat %>% apply(2,quantile,probs=c(0.25))]
    pred.stat.tbl[geoid==gid, qtile75 := tmp.pred.mat %>% apply(2,quantile,probs=c(0.75))]
  }
  
  return(pred.stat.tbl)
}





#' Generate Posterior Mean Predictions
#'
#' @param in.beta.tbl - data table of posterior beta simulations
#' @param in.covid.tbl - data table of covid training data
#'
#' @return plot of posterior mode and mean predictions
gen.posterior.preds <- function(in.beta.tbl,in.covid.tbl) {
  
  # plot posterior predictions
  # beta.mean.tbl <- in.beta.tbl[,.SD %>% lapply(mean),by=geoid,.SDcols=c("b0","b1","b2")]
  # in.covid.tbl <- in.covid.tbl %>% merge(beta.mean.tbl,by = "geoid", all.x = T)
  # in.covid.tbl[,log.cases.pred := b0 + (b1*date.int.delta) + (b2*(date.int.delta^2)) ]
  
  pred.tbl <- compute.predictions(in.covid.tbl,in.beta.tbl)
  in.covid.tbl <- in.covid.tbl %>% merge(pred.tbl,by = c("geoid","date.int.delta"), all.x = T)
  
  plt.tbl <- in.covid.tbl[,.(country,geoid,date,adj.log.cases,mean,mode,median,qtile25,qtile75)]
  plt.tbl <- plt.tbl %>% melt(measure.vars=c("mean","mode","median","qtile25","qtile75"), variable.name = "statistic")
  
  ggplot(plt.tbl) +
    geom_point(aes(x=date,y=adj.log.cases)) +
    geom_line(aes(x=date,y=value,color=statistic),size=0.5) +
    facet_wrap(~geoid) +
    scale_x_date(date_breaks = "5 days") +
    theme(axis.text.x = element_text(angle = 90))
  
}




