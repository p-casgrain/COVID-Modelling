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
    
    ggplot(plt.dt) +
      geom_density(aes(x=value.date, group=interaction(variable,geoid)), 
                   fill="purple", alpha=0.5, colour = 'black', size = 0.25 ) +
      geom_vline(aes(colour="Mean", xintercept=value.date.mean ), size=1) +
      geom_vline(aes(colour="Median", xintercept=value.date.median ),  size=1) +
      geom_vline(aes(colour="Mode", xintercept=value.date.mode ),  size=1) +
      ggtitle("Peak Date and End Date - Probability Densities") +
      theme(axis.text.x = element_text(angle = 90)) +
      facet_grid(variable~geoid, scales="free_y") +
      scale_x_date(date_breaks = "15 days")
  }


#' Generate Posterior Mean Predictions
#'
#' @param in.beta.tbl - data table of posterior beta simulations
#' @param in.covid.tbl - data table of covid training data
#'
#' @return plot of posterior mode and mean predictions
gen.posterior.preds <- function(in.beta.tbl,in.covid.tbl) {
  
  # plot posterior predictions
  beta.mean.tbl <- in.beta.tbl[,.SD %>% lapply(mean),by=geoid,.SDcols=c("b0","b1","b2")]
  in.covid.tbl <- in.covid.tbl %>% merge(beta.mean.tbl,by = "geoid", all.x = T)
  in.covid.tbl[,log.cases.pred := b0 + (b1*date.int.delta) + (b2*(date.int.delta^2)) ]
  
  
  ggplot(in.covid.tbl) +
    geom_point(aes(x=date,y=adj.log.cases)) +
    geom_line(aes(x=date,y=log.cases.pred), color="red") +
    facet_wrap(~geoid) +
    scale_x_date(date_breaks = "5 days") +
    theme(axis.text.x = element_text(angle = 90))
  
}




