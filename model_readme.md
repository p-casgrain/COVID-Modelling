# COVID-19 Model Spec Sheet

## The Data

This GitHub repo contains various poorly documented tools for modelling and manipulating [data on the global geographic distribution of of COVID-19 cases](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide) provided by the *European Center for Disease Control* (ECDC).

## The Model

We model the $\log$ of the number of new cases per day in each country, based on the ECDC dataset. We employ a relatively simple latent-regression approach which is described over the course of this section.

Since the disease is in different stages of progression across countries, it is desirable to global information to model disease progression in individual countries. The approach we take a latent modelling approach in order to achieve this effect.

Let the set of indices $j\in\mathfrak{N}$ represent individual countries and $i\in\mathfrak{M}$ represent single data points within each countries. Thus, each row in out dataset is indexed by a tuple $(j,i)$. Define 

\begin{align}
<!--\tau_{j,i} &:= \text{# of days since first case in country-$j$ for datapoint $(j,i)$} \,, \\-->
<!--y_{j,i} &:= \log(\text{# of new cases on day for $(j,i)$}) \,, \\
-->
<!--x_{j,i} &:= [1,\tau_{j,i},\tau_{j,i}^2] \,.-->
\end{align}

Our model can be summarised as 
$$
y_{i,j} = 
\langle x_{i,j} \,,\, \beta_j \rangle + 
\sigma_j \,\epsilon_{i,j} 
\;,\;\; 
\epsilon_{i,j} \sim \text{i.i.d. } \mathcal{N}(0,1)
\;,\;\;
\\beta_{j} \sim \text{i.i.d. } \mathcal{N}_{\Omega}( \mu , \Sigma)
$$
where $\sigma_j>0$, $\mu \in \mathbb{R}^3$ and $\Sigma\in\mathbb{R}^{3\times 3}$ is positive-definite and $\Omega\subseteq\mathbb{R}^3$ is convex. We define the distribution of $\mathcal{N}_{\Omega}$ to be a Gaussian random variable truncated to the set $\Omega$.

<!--## Estimating Model Parameters

## Prediction -->

