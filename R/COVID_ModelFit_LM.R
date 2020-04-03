source("./COVID_DataGrab.R")



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
             lower.limits = c(0,-Inf,-Inf),
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
# coef.covmat.inv <- coef.covmat %>% Inver
coef.mean <- model.coefs.tbl[,.(b0,b1,b2) %>% sapply(mean)]


get.quad.max <- ( function(cfs) -cfs["1"]/(2*cfs["2"]) ) %>% Vectorize
get.quad.roots <- ( function(cfs) ( -cfs["1"] + c(-1,1)*sqrt(cfs["1"]^2 - 4*cfs["0"]*cfs["2"]) )/(2*cfs["2"]) ) %>% Vectorize
get.quad.root.gap <- ( function(cfs) ( sqrt(cfs["1"]^2 - 4*cfs["0"]*cfs["2"]) )/cfs["2"] ) %>% Vectorize


model.coefs %>% get.quad.max
model.coefs %>% get.quad.roots
model.coefs %>% get.quad.root.gap





ggplot(covid.tbl[geoid=="CN"]) +
  geom_point(aes(x=date,y=cases,colour=geoid))


