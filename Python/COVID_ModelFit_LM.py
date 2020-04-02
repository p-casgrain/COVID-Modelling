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

# Import DataGrab Functions
from COVID_DataGrab import *

# Import Model Fitting Functions
from collections import *
import torch


def camelCase(st):
    output = ''.join(x for x in st.title() if x.isalnum())
    return output[0].lower() + output[1:]


def negPart(x):
    return -1*torch.relu(-1*x)


def posPart(x):
    return torch.relu(x)


def torchCov(x):
    return torch.matmul(x.T, x)/x.shape[0]


def projectBeta(beta):
    beta[:, 0] = negPart(beta[:, 2])
    beta[:, 1] = posPart(beta[:, 2])
    beta[:, 2] = negPart(beta[:, 2])
    return beta

# Define Log-Density Functions


def log_density_country(x, y, beta, rho):
    n = len(y)
    prec = rho**2
    norm_const = torch.tensor(2*np.pi, requires_grad=False)
    return - 0.5*torch.norm(torch.mv(x, beta.double()) - y)*prec \
        + 0.5*n*torch.log(prec) - 0.5*n*torch.log(norm_const)


def log_density_prior(beta, mean, precChol):
    prec = torch.matmul(precChol.T, precChol)
    beta_adj = (beta - mean).unsqueeze(-1)
    return - 0.5*torch.chain_matmul(beta_adj.T, prec, beta_adj) \
        + 0.5*torch.logdet(prec/2*np.pi)


class bayesLR(object):
    def __init__(self, x_train, y_train, initParamDct):
        # Assign training data and initialize parameters
        self.x_train = \
            [torch.tensor(it, dtype=torch.float64, requires_grad=False)
             for it in x_train]
        self.y_train = \
            [torch.tensor(it, dtype=torch.float64, requires_grad=False)
             for it in y_train]
        self.paramDct = \
            {k: torch.tensor(v, dtype=torch.float64, requires_grad=True)
             for k, v in initParamDct.items()}
        self.n_ctrys = len(self.y_train)

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

    def EM_step(self):
        # Run a single step of the EM algorithm
        with torch.no_grad():  # Ensure computations aren't tracked
            paramDct0 = {k: v.clone() for (k, v) in self.paramDct.items()}
            # Update Parameters for Prior Distribution on Coefficients
            self.paramDct['priorMean'] = \
                self.paramDct0['ctryBetas'].mean(dim=0)
            self.priorPrec = torchCov(self.paramDct0['ctryBetas']).pinverse()
            self.paramDct['priorPrecChol'] = self.priorPrec.cholesky()
            # Update per-country parameters
            for ix in range(self.n_ctrys):
                # Generate predictions, residuals and covariances
                ctryPred = torch.mv(self.x_train[ix],
                                    self.paramDct0['ctryBetas'][ix].double())
                ctryResid = ctryPred - self.y_train[ix]
                # Compute residual error precision
                self.paramDct['errorPrec'] = 1.0/ctryResid.var()
                # Compute per-country betas
                betaXY_tmp = (self.ctryXY[ix] * paramDct0['errorPrec']) + \
                    torch.mv(self.priorPrec, self.paramDct0['priorMean'])
                betaXX_tmp = self.ctryXX[ix] * paramDct0['errorPrec'] + \
                    self.priorPrec
                self.paramDct['ctryBetas'] = \
                    torch.solve(betaXY_tmp,
                                betaXX_tmp).solution.squeeze()

    def update_likelihood(self):
        self.log_likelihood = 0
        for ix in range(len(train_geoid_lst)):
            self.log_likelihood += \
                self.log_density_country(self.x_train[ix],
                                         self.y_train[ix],
                                         self.paramDct['ctryBetas'][ix],
                                         self.paramDct['errorPrec'][ix])
            self.log_likelihood += \
                self.log_density_prior(self.paramDct['ctryBetas'][ix],
                                       self.paramDct['priorMean'],
                                       self.paramDct['priorPrecChol'])

    def log_density_country(self, x, y, beta, rho):
        # Compute log density of data for an individual country
        n_obs = len(y)
        prec = rho.abs()
        yhat = torch.mv(x, beta.double())
        norm_const = 2*np.pi
        return -0.5*(torch.norm(yhat - y)*prec -
            n_obs*torch.log(prec/norm_const))

    def log_density_prior(self, beta, mean, precChol):
        # Compute log density of prior portion of likelihood
        prec = torch.matmul(precChol.T, precChol)
        beta_adj = (beta - mean).unsqueeze(-1)
        norm_const = 2*np.pi
        return - 0.5*(torch.chain_matmul(beta_adj.T, prec, beta_adj)
                      - torch.logdet(prec/2*np.pi)))


class expDecayLrnRate(object):
    def __init__(self, l0, lMin, lDecay):
        self.lMin=lMin
        self.lDecay=lDecay
        self.l=max(l0, self.lMin)

    def next(self):
        self.l=max(self.l*self.lDecay, self.lMin)
        return self.l

    def __mul__(self, other):
        return(self.l*other)


if __name__ == '__main__':

    # Try to Download Latest Data to Directory
    covid_csv_root=os.path.expanduser(
        '~/Documents/GitHub/COVID-Experiments/CSV/')

    # Load Latest enriched CSV file generated by R
    covid=covid_enriched_load_latest(covid_csv_root)
    covid=covid.rename(dict([(x, camelCase(x))
                               for x in covid.columns]), axis = 'columns')

    # Subset to Training Data
    def train_subset(x): return (x.cases > 0) & (x.totCases > 1000) &\
        (x.nNonzeroRows > 20) & (x.maxGap < 4) &\
        (x.geoid != "RU")
    covid_train=covid[train_subset(covid)]

    # Generate Arrays of Training Data
    def genFeatures(x):
        y=torch.tensor(
            (x.dateInt - x.ctryMinDateInt).to_numpy(),
            dtype = torch.float64, requires_grad = False)
        return torch.stack((y**0, y**1, y**2)).T

    def genResp(x):
        return torch.tensor(np.log(x.cases.to_numpy()),
                            dtype = torch.float64, requires_grad = False)

    train_geoid_lst=covid_train.geoid.unique()
    n_geoid=len(train_geoid_lst)

    x_train=[covid_train[covid_train.geoid == id].pipe(
        genFeatures) for id in train_geoid_lst]
    y_train=[covid_train[covid_train.geoid == id].pipe(
        genResp) for id in train_geoid_lst]

    # Initialize Beta Parameters
    ctryBetas=torch.randn((n_geoid, 3),
                    dtype=torch.float64, requires_grad=True)

    with torch.no_grad():
        for i in range(n_geoid):
            ctryBetas[i, ] = \
                torch.solve(torch.matmul(x_train[i].T, y_train[i]).unsqueeze(-1),
                            torch.matmul(x_train[i].T, x_train[i])).solution.squeeze()
        priorMean = ctryBetas.mean(0)

    priorMean.requires_grad_(True)

    # Initialize Precision Parameters
    errorPrec = \
        torch.ones((n_geoid, 1),
                   dtype=torch.float64, requires_grad=True)

    priorPrecChol = torch.eye(3, dtype=torch.float64, requires_grad=True)

    # Set learning rate params
    lrn_rate = expDecayLrnRate(l0=1e-10, lMin=1e-10, lDecay=1-1e-5)

    # Train the parameters
    n_iter = 5
    print_interval = 300

    for it_num in range(n_iter):
        # Compute Likelihood
        log_lik = 0
        for ix in range(len(train_geoid_lst)):
            log_lik += \
                log_density_country(x_train[ix], y_train[ix],
                                    ctryBetas[ix], errorPrec[ix]) \
                + log_density_prior(ctryBetas[ix], priorMean, priorPrecChol)

        # Print progress
        if not it_num % print_interval:
            print(it_num, "Loss: %f" % log_lik.detach())

        # Compute Gradient and take step
        log_lik.backward()
        with torch.no_grad():
            # Compute Gradients
            ctryBetas += lrn_rate * ctryBetas.grad
            ctryBetas = projectBeta(ctryBetas)

            priorMean += lrn_rate * priorMean.grad
            priorMean.data = projectBeta(priorMean.unsqueeze(0)).squeeze().data

            priorPrecChol += lrn_rate * priorPrecChol.grad

            errorPrec += lrn_rate * errorPrec.grad
            errorPrec.data = torch.abs(errorPrec.data).data

        lrn_rate.next()

        for it_num in range(n_iter):
            # Compute Likelihood
        log_lik = 0
        for ix in range(len(train_geoid_lst)):
            log_lik += \
                log_density_country(x_train[ix], y_train[ix],
                                    ctryBetas[ix], errorPrec[ix]) \
                + log_density_prior(ctryBetas[ix], priorMean, priorPrecChol)

        # Print progress
        if not it_num % print_interval:
            print(it_num, "Loss: %f" % log_lik.detach())

        # Compute Gradient and take step
        log_lik.backward()
        with torch.no_grad():
            # Compute Gradients
            ctryBetas += lrn_rate * ctryBetas.grad
            ctryBetas = projectBeta(ctryBetas)

            priorMean += lrn_rate * priorMean.grad
            priorMean.data = projectBeta(priorMean.unsqueeze(0)).squeeze().data

            priorPrecChol += lrn_rate * priorPrecChol.grad

            errorPrec += lrn_rate * errorPrec.grad
            errorPrec.data = torch.abs(errorPrec.data).data

        lrn_rate.next()

    for it_num in range(n_iter):
        pred =

    # Save Latest Params to Disk
    param_fname = "covid_lm_params_" \
        + dtm.datetime.now().strftime("%Y%m%d_%H%m")\
        + ".pickle"

    param_dir = os.path.expanduser(
        '~/Documents/GitHub/COVID-Experiments/ParamPickles/')

    param_write_path = os.path.join(param_dir, param_fname)

    with open(param_write_path, "wb") as handle:
        pickle.dump({'ctryBetas': ctryBetas,
                     'errorPrec': errorPrec,
                     'priorMean': priorMean,
                     'priorPrecChol': priorPrecChol},
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Generate Model Predictions
