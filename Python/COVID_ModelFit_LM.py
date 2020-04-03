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

def torchOuterSum(x):
    return torch.matmul(x.T, x)/x.shape[0]

def torchCov(x,mn):
    mn = mn.squeeze().unsqueeze(0)
    return torchOuterSum(x-mn)

def projectBeta(beta):
    # beta[:, 0] = negPart(beta[:, 0])
    # beta[:, 1] = posPart(beta[:, 1])
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


class gaussBayesLM(object):
    def __init__(self, x_train, y_train, initParamDct, re_normalize = False):
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

        # Re-normalize features if approptiate
        if re_normalize:
            self.x_adj = [self.gen_x_adj(it) for it in self.x_train]
            self.x_train = \
                [torch.matmul(self.x_train[ix],
                self.x_adj[ix]) for ix in range(self.n_ctrys)]

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

    def gen_x_adj(self,x):
        x_adj = torch.tensor( [ 1.0 if it==0 else 1/it for it in x.std(dim=0) ] )
        return torch.diag(x_adj)

    def EM_step(self):
        # Run a single step of the EM algorithm
        with torch.no_grad():  # Ensure computations aren't tracked
            # Copy latest parameters
            self.paramDct0 = {k: v.clone().detach() for (k, v) in self.paramDct.items()}

            # Update Parameters for Prior Distribution on Coefficients
            self.paramDct['priorMean'] =  self.paramDct0['ctryBetas'].mean(dim=0)

            self.priorCov = \
                torchOuterSum(self.paramDct0['ctryBetas'] \
                    - self.paramDct0['priorMean'])

            self.priorPrec = self.priorCov.inverse()
            self.paramDct['priorPrecChol']= self.priorPrec.cholesky()

            # Update per-country parameters
            for ix in range(self.n_ctrys):
                # Generate predictions, residuals and covariances
                ctryPred=torch.mv(self.x_train[ix],
                                     self.paramDct0['ctryBetas'][ix].double())
                ctryResid=ctryPred - self.y_train[ix]

                # Compute residual error precision
                self.paramDct['errorPrec'][ix]=1.0/ctryResid.var()

                # Compute per-country betas
                self.betaXY_tmp=\
                    (self.ctryXY[ix] *  self.paramDct0['errorPrec'][ix]).T \
                        + torch.mv(self.priorPrec,  self.paramDct0['priorMean'])
                self.betaXX_tmp=\
                    self.ctryXX[ix] *  self.paramDct0['errorPrec'][ix] \
                        + self.priorPrec
                self.paramDct['ctryBetas'][ix]=\
                    torch.solve(self.betaXY_tmp.T,self.betaXX_tmp)\
                        .solution.squeeze()

    def update_likelihood(self):
        self.log_likelihood= 0
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
        return -0.5*(torch.norm(yhat - y)*prec -
            n_obs*torch.log(prec/norm_const))

    def log_density_prior(self, beta, mean, precChol):
        # Compute log density of prior portion of likelihood
        prec = torch.matmul(precChol.T, precChol)
        beta_adj = (beta - mean).unsqueeze(-1)
        norm_const = 2*np.pi
        return - 0.5*(torch.chain_matmul(beta_adj.T, prec, beta_adj)
                      - torch.logdet(prec/2*np.pi))


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

    covid_train=covid[covid.isTrain==True]

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

    # Initialize Model Parameters
    ctryBetas=torch.randn((n_geoid, 3),dtype=torch.float64)

    for i in range(n_geoid):
        ctryBetas[i, ] = \
            torch.solve(torch.matmul(x_train[i].T, y_train[i]).unsqueeze(-1),
                        torch.matmul(x_train[i].T, x_train[i])).solution.squeeze()
    
    priorMean = ctryBetas.mean(0)

    errorPrec = torch.ones((n_geoid, 1),dtype=torch.float64)

    priorPrecChol = torch.eye(3, dtype=torch.float64)

    paramDctInit = {'ctryBetas':ctryBetas,
                    'priorMean':priorMean,
                    'errorPrec':errorPrec,
                    'priorPrecChol':priorPrecChol}

    # Initialize Model Object, then delete parameters
    ctryLM = gaussBayesLM(x_train, y_train, paramDctInit,re_normalize=False)

    # Set learning rate params
    lrn_rate = expDecayLrnRate(l0=25e-7, lMin=8e-8, lDecay=1-2e-4)

    # Train the parameters
    n_iter = 50000
    print_interval = 250

    for it_num in range(n_iter):
        # Update Likelihood
        ctryLM.update_likelihood()

        # Update Learning Rate
        lrn_rate.next()

        # Take EM step
        # ctryLM.EM_step()

        # Compute Gradient and take step
        ctryLM.log_likelihood.backward()

        with torch.no_grad():
            # Compute Gradients
            ctryLM.paramDct = \
                {k: (v + lrn_rate*v.grad).requires_grad_(True) \
                    for k,v in ctryLM.paramDct.items()}
            
            ctryLM.paramDct['ctryBetas'].data = \
                projectBeta(ctryLM.paramDct['ctryBetas'].data)
                
            ctryLM.paramDct['priorMean'].data = \
                projectBeta(ctryLM.paramDct['priorMean']
                            .data.unsqueeze(0)).squeeze()
            
            ctryLM.paramDct['errorPrec'].data = \
                torch.abs(ctryLM.paramDct['errorPrec'].data)

        # Print progress
        if not (it_num+1) % print_interval:
            print(it_num+1, 
            "Loss: %f" % ctryLM.log_likelihood, 
            "Log10LrnRt: %f" % np.log10(lrn_rate.l) )


    # Save Latest Params to Disk
    param_fname = "covid_lm_params_" \
        + dtm.datetime.now().strftime("%Y%m%d_%H%m")\
        + ".pickle"

    param_dir = os.path.expanduser(
        '~/Documents/GitHub/COVID-Experiments/ParamPickles/')

    param_write_path = os.path.join(param_dir, param_fname)

    with open(param_write_path, "wb") as handle:
        pickle.dump(ctryLM.paramDct,
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Generate Model Predictions
