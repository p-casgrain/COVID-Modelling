# DataGrab Dependencies (not sure how to skip this)
import os
import re
import pandas as pd
import numpy as np
from datetime import date, timedelta
import datetime as dtm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import itertools as iter
# from plotnine import ggplot, geom_point, geom_density, aes, stat_smooth, facet_wrap


# Import DataGrab Functions
from COVID_DataGrab import *

# Import Model Fitting Functions
from collections import *
import torch
from torch.distributions import Normal, MultivariateNormal

# Define Utility Functions


def camelCase(st):
    output = ''.join(x for x in st.title() if x.isalnum())
    return output[0].lower() + output[1:]


def projectHalfSpace(x, sides=None, lower=None):
    if sides == None:
        sides = np.repeat(1.0, x.shape[1])
    if lower == None:
        lower = np.repeat(0.0, x.shape[1])
    for ix in range(len(sides)):
        lower = np.abs(lower)
        if not sides[ix] == None:
            x[:, ix] = sides[ix] * \
                (lower[ix] + torch.relu(sides[ix]*x[:, ix] - lower[ix]))
    return x


def sgnRelu(x, s):
    return s*torch.relu(x*s)


def gen_initParamDct(x_train, y_train, n_geoid):
    ctryBetas = torch.randn((n_geoid, 3), dtype=torch.float64)

    for i in range(n_geoid):
        ctryBetas[i, ] = \
            torch.solve(torch.matmul(x_train[i].T, y_train[i]).unsqueeze(-1),
                        torch.matmul(x_train[i].T, x_train[i])).solution.squeeze()

    priorMean = ctryBetas.mean(0)

    errorPrec = torch.ones((n_geoid, 1), dtype=torch.float64)

    priorPrecChol = torch.eye(3, dtype=torch.float64)

    paramDctInit = {'ctryBetas': ctryBetas,
                    'priorMean': priorMean,
                    'errorPrec': errorPrec,
                    'priorPrecChol': priorPrecChol}

    return paramDctInit


# Define model and training rate classes

class gaussBayesLM(object):
    def __init__(self, x_train, y_train, initParamDct, re_normalize=False):
        # Assign training data and initialize parameters
        self.x_train = \
            [it.clone().detach().requires_grad_(False)
             for it in x_train]
        self.y_train = \
            [it.clone().detach().clone().detach().requires_grad_(False)
             for it in y_train]
        self.paramDct = \
            {k: v.clone().detach().clone().detach().requires_grad_(True)
             for k, v in initParamDct.items()}
        self.n_ctrys = len(self.y_train)

        # Re-normalize features if appropriate
        if re_normalize:
            self.x_adj = [self.gen_x_adj(it) for it in self.x_train]

            self.x_train = \
                [torch.matmul(self.x_train[ix],
                              self.x_adj[ix]) for ix in range(self.n_ctrys)]

            with torch.no_grad():
                for ix in range(self.n_ctrys):
                    self.paramDct['ctryBetas'][ix, ] = \
                        torch.matmul(self.paramDct['ctryBetas'][ix, ],
                                     self.x_adj[ix].inverse())
                self.paramDct['priorMean'] = self.paramDct['ctryBetas'].mean(0)

        self.x_normalized = re_normalize

        # Compute Covariance Matrices
        self.ctryXX = \
            [torch.matmul(self.x_train[ix].T,
                          self.x_train[ix]) / self.x_train[ix].shape[0]
             for ix in range(self.n_ctrys)]
        self.ctryXY = \
            [torch.matmul(x_train[ix].T,
                          y_train[ix]).unsqueeze(-1) / self.x_train[ix].shape[0]
             for ix in range(self.n_ctrys)]

        # Update likelihood for First time
        self.update_likelihood()

    def gen_x_adj(self, x):
        x_adj = torch.tensor([1.0 if it == 0 else 1/it for it in x.std(dim=0)])
        return torch.diag(x_adj)

    def update_likelihood(self):
        self.log_likelihood = 0
        for ix in range(self.n_ctrys):
            self.log_likelihood += \
                self.log_density_country(self.x_train[ix],
                                         self.y_train[ix],
                                         self.paramDct['ctryBetas'][ix],
                                         self.paramDct['errorPrec'][ix]).squeeze()
            self.log_likelihood += \
                self.log_density_prior(self.paramDct['ctryBetas'][ix],
                                       self.paramDct['priorMean'],
                                       self.paramDct['priorPrecChol']).squeeze()

    def log_density_country(self, x, y, beta, rho):
        # Compute log density of data for an individual country
        n_obs = len(y)
        prec = rho.abs()
        yhat = torch.mv(x, beta.double())
        norm_const = 2*np.pi
        return -0.5*((yhat - y).pow(2).sum()*prec - n_obs*torch.log(prec/norm_const))

    def log_density_prior(self, beta, mean, precChol):
        # Compute log density of prior portion of likelihood
        prec = torch.matmul(precChol.T, precChol)
        beta_adj = (beta - mean).unsqueeze(-1)
        norm_const = 2*np.pi
        return - 0.5*(torch.chain_matmul(beta_adj.T, prec, beta_adj)
                      - torch.logdet(prec/norm_const))

    def update_posterior_distributions(self):
        self.post_mean = []
        self.post_prec = []
        self.post_cov = []
        self.post_distributions = []
        with torch.no_grad():
            # Compute re-used portion
            tmp_priorPrecInner = \
                torch.matmul(self.paramDct['priorPrecChol'].T,
                             self.paramDct['priorPrecChol'])
            tmp_priorPrecMnMv = \
                torch.mv(tmp_priorPrecInner,
                         self.paramDct['priorMean']).unsqueeze(-1)

            # Compute per-country posteriors
            for ix in range(self.n_ctrys):
                # Compute Posterior Precision & Covariance
                self.post_prec.append(
                    (self.paramDct['errorPrec'][ix] * self.ctryXX[ix]) +
                    tmp_priorPrecInner)

                self.post_cov.append(self.post_prec[ix].pinverse())

                # Compute Posterior Mean using Precision
                self.post_mean.append(
                    (self.ctryXY[ix] * self.paramDct['errorPrec'][ix]) +
                    tmp_priorPrecMnMv
                )

                self.post_mean[ix] = \
                    torch.mv(self.post_cov[ix],
                             self.post_mean[ix].squeeze())

                # Update Distribution Objects
                self.post_distributions.append(
                    MultivariateNormal(self.post_mean[ix],
                                       self.post_cov[ix]))

    def sample_posterior(self, n):
        self.update_posterior_distributions()
        return [d.sample((n,)) for d in self.post_distributions]

    def predict(self, x_new, beta_lst=None):
        if beta_lst == None:
            beta_lst = self.sample_posterior(5000)
        # Renormalize features if necessary
        if self.x_normalized:
            x_new = \
                [torch.matmul(x_new[ix],
                              self.x_adj[ix]) for ix in range(self.n_ctrys)]

        # Return predictions
        return [torch.matmul(x_new[ix],
                             beta_lst[ix].T) for ix in range(self.n_ctrys)]


class expDecayLrnRate(object):
    def __init__(self, l0, lMin, lDecay):
        self.lMin = lMin
        self.lDecay = lDecay
        self.l = max(l0, self.lMin)

    def next(self):
        self.l = max(self.l*self.lDecay, self.lMin)
        return self.l

    def __mul__(self, other):
        return(self.l*other)


class lipsEstimLrnRate(object):
    def __init__(self, l0, l_min,  estimStopIter=int(1e3)):
        self.n = 0
        self.l = l0
        self.l_min = l_min
        self.estimStopIter = estimStopIter
        self.x_last = None
        self.g_last = None

    def next(self, x, g):
        # Step forward integer clock
        self.n += 1
        # Try to estimate inverse lipschitz constant
        if (self.n < self.estimStopIter):
            if not ((self.x_last == None) | (self.g_last == None)):
                if (self.x_last - x).norm().item()>0:
                    self.l = \
                        min(self.l,
                            (x-self.x_last).norm().item() / (g-self.g_last).norm().item())
            self.l = max(self.l, self.l_min)
            self.x_last = x
            self.g_last = g
            return self.l

    def __mul__(self, other):
        return(self.l*other)


if __name__ == '__main__':

    # Try to Download Latest Data to Directory
    covid_csv_root = os.path.expanduser(
        '~/Documents/GitHub/COVID-Experiments/CSV/')

    # Load Latest enriched CSV file generated by R
    covid = covid_enriched_load_latest(covid_csv_root)
    covid = covid.rename(dict([(x, camelCase(x))
                               for x in covid.columns]), axis='columns')
    covid['date'] = pd.to_datetime(covid.date, format="%Y-%m-%d")

    covid = covid.sort_values(by=['geoid', 'dateInt'])

    covid_train = covid.loc[covid.isTrain == True]

    # Generate Arrays of Training Data
    def genFeatures(x):
        y = torch.tensor(x.dateIntDelta.to_numpy(),
            dtype=torch.float64, requires_grad=False)
        return torch.stack((y**0, y**1, y**2)).T

    def genResp(x):
        return torch.tensor(x.adjLogCases.to_numpy(),
                            dtype=torch.float64, requires_grad=True)

    train_geoid_lst = covid_train.geoid.unique()
    n_geoid = len(train_geoid_lst)

    x_train = [covid_train[covid_train.geoid == id].pipe(
        genFeatures) for id in train_geoid_lst]

    y_train = [covid_train[covid_train.geoid == id].pipe(
        genResp) for id in train_geoid_lst]

    # Initialize Model Parameters from Disk
    # param_read_path = "/Users/philippe/Documents/GitHub/COVID-Experiments/ParamPickles//Users/philippe/Documents/GitHub/COVID-Experiments/ParamPickles/covid_lm_params_20200406_2109.pickle"
    # with open(param_read_path, "rb") as h:
    #     paramDctInit = pickle.load(h)

    paramDctInit = \
        gen_initParamDct(x_train,y_train,n_geoid)

    # Initialize Model Object, then delete parameters
    ctryLM = gaussBayesLM(x_train, y_train, paramDctInit, re_normalize=False)

    # Initiailze learning rates
    lrn_rate_dct = \
        {k: lipsEstimLrnRate(5e-4, 1e-9, estimStopIter=1500)
            for k in ctryLM.paramDct.keys()}

    # === Train the Model ===
    n_iter = int(1000)
    print_interval = 250

    for it_num in range(n_iter):
        # Update Likelihood
        ctryLM.update_likelihood()

        # Compute Gradient and take step
        ctryLM.log_likelihood.backward()

        with torch.no_grad():
            # Update Learning Rates
            for k, v in lrn_rate_dct.items():
                v.next(ctryLM.paramDct[k],
                       ctryLM.paramDct[k].grad)

            # Compute Gradients
            ctryLM.paramDct = \
                {k: (v + lrn_rate_dct[k].l*v.grad).requires_grad_(True)
                    for k, v in ctryLM.paramDct.items()}

            ctryLM.paramDct['ctryBetas'].data[:, 2] = \
                sgnRelu(ctryLM.paramDct['ctryBetas'].data[:, 2], -1.0)
            
            ctryLM.paramDct['ctryBetas'].data[:, 1] = \
                torch.relu(ctryLM.paramDct['ctryBetas'].data[:, 1])


            ctryLM.paramDct['priorMean'].data[2] = \
                sgnRelu(ctryLM.paramDct['priorMean'].data[2], -1.0)

            ctryLM.paramDct['errorPrec'].data = \
                torch.abs(ctryLM.paramDct['errorPrec'].data)

        # Print progress
        if not (it_num+1) % print_interval:
            print(it_num+1,
                  "Loss: %f" % ctryLM.log_likelihood, "\n"
                  "Log10LrnRt: %s" % {k: np.log10(v.l) for k, v in lrn_rate_dct.items()})

    # Save Latest Params to Disk
    try:
        param_fname = "covid_lm_params_" \
            + dtm.datetime.now().strftime("%Y%m%d_%H%M")\
            + ".pickle"

        param_dir = os.path.expanduser(
            '~/Documents/GitHub/COVID-Experiments/ParamPickles/')

        param_write_path = os.path.join(param_dir, param_fname)

        with open(param_write_path, "wb") as handle:
            pickle.dump(ctryLM.paramDct.copy(),
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Params written to %s." % param_write_path)
    except:
        print("Params could not be written.")
    # === Generate Model Predictions ===

    # Use rejection sampling to obtain posterior beta samples

    def accept_quad(x):
        # Return True if:
        # (1) curvature negative
        # (2) real roots exist
        # (3) peak happens happens after 0
        c1 = x[:, 1].pow(2) - (4*x[:, 0]*x[:, 2])
        return (x[:, 2] < 0) & (c1 > 0) & (x[:, 1] > 0)

    n_samples = int(1e4)
    post_beta_samples = ctryLM.sample_posterior(n_samples)
    post_beta_samples = [el[accept_quad(el), ].float() for el in post_beta_samples]
    post_beta_samples = dict(zip(train_geoid_lst, post_beta_samples))

    # Write Samples to Disk
    post_beta_samples_df = \
        pd.concat([pd.DataFrame({'geoid': k, 'b0': v[:, 0],
                                 'b1':v[:, 1], 'b2':v[:, 2]})
                   for k, v in post_beta_samples.items()])

    post_sim_fname = "covid_lm_params_" \
        + dtm.datetime.now().strftime("%Y%m%d_%H%M")\
        + ".csv"
    post_sim_dir = os.path.expanduser(
        '~/Documents/GitHub/COVID-Experiments/Post_Sims/')
    param_write_path = os.path.join(post_sim_dir, post_sim_fname)

    post_beta_samples_df.to_csv(param_write_path)

    # Compute posterior means

    post_mean_betas = {k: v.mean(0) for k, v in post_beta_samples.items()}

    # Estimate mean peak time
    def cpt_peak_time(x):
        return (x[:, 1]/(-2*x[:, 2]))

    peak_times = \
        {k: cpt_peak_time(v).numpy() for k, v in post_beta_samples.items()}

    peak_times_mean = {k: v.mean() for k, v in peak_times.items()}

    # Get mean polynomial roots
    def cpt_quad_roots(x):
        c0 = -x[:, 1]
        c1 = torch.sqrt(x[:, 1].pow(2) - (4*x[:, 0]*x[:, 2]))
        c2 = 2*x[:, 2]
        return torch.stack([(c0 - c1)/c2, (c0 + c1)/c2], dim=1)

    quad_roots = \
        {k:cpt_quad_roots(v).sort().values \
            for k,v in post_beta_samples.items()}

    quad_roots_mean = \
        {k: v.mean(0).numpy() for k, v in quad_roots.items()}

    def addDays(t0, deltaDays):
        try:
            return t0 + dtm.timedelta(days=deltaDays)
        except:
            print(t0, deltaDays)
            raise ValueError()

    def convert_tau_to_date(tau, ctry_geoid, tbl):
        initDate = \
            (tbl
             .loc[tbl.geoid == ctry_geoid]
             .loc[tbl.ctryMinDateInt == tbl.dateInt]
             .date.iloc[0].date())
        if (type(tau) == list):
            return [addDays(initDate, min(t, int(730))) for t in tau]
        else:
            return addDays(initDate, tau)

    peak_times_date = \
        {k:convert_tau_to_date(v.tolist(),k,covid) \
            for k,v in peak_times_mean.items()}

    peak_dt = \
        ( list(zip(iter.repeat(k), v)) \
          for k, v in peak_times.items())

    # peak_dt = iter.chain.from_iterable(peak_dt)
    # peak_dt = pd.DataFrame.from_records(peak_dt)
    # peak_dt = peak_dt.rename(columns={0: 'geoid', 1: 'peakDate'})

    # ordered_days = peak_dt.geoid
    # g = sns.FacetGrid(peak_dt, row="geoid",
    #                   height=1.7, aspect=4, sharex=False)
    # g.map(sns.distplot, "peakDate", hist=False, rug=True)
    # plt.show()

    # pltArray = peak_dt.loc[peak_dt.geoid == "CA"].peakDate
    # pltArray = [np.log(x) for x in pltArray.to_numpy()]
    # pltArray.sort()

    # ixs = (len(pltArray) *
    #        np.array([0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])).round()

    # [pltArray[int(ix-1)] for ix in ixs]

    # sns.distplot(pltArray, hist=False, rug=False)
