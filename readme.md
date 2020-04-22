# COVID19 Temporal Evolution Model
**by Philippe Casgrain**

### How to Use

Run `<root>/R/COVID_MainScript.R`, and adjust root project path accordingly.

### Data Used

The model uses the number of new COVID-19 cases per day, per country as [published daily by the European Center for Disease Contril](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide). Latest data is automatically downloaded by the included scripts each day and included in each iteration of model fitting.

### Brief Model Description

This uses a simple latent quadratic regression model to predict the number of new cases per day. Each country's new cases per day curve is modeled as `log(cases) = f(t) + e_t =  b0 + b1 * t + b2 * t^2 + e_t`, where `t` is the integer date, `e_t` is scaled Gaussian white noise and `(b0,b1,b2)` are latent and drawn from a truncated multivariate Gaussian distribution for each country. The truncation is done to ensure that `f"(t) < 0` and `f(t)` has two distinct real positive roots with probability one.

The fitting procedure estimates the parameters of the truncated Gaussian via marginal likelihood maximization across all countries simultaneously. Using these parameters, random samples are drawn from the posterior distribution of `(b0,b1,b2)` using rejection sampling, and these samples are then used to approximate the densities of the date for both the *peak-new-cases-per-day* and *date of the last new case* for each individual country.