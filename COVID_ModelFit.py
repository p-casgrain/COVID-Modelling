# DataGrab Dependencies (not sure how to skip this)
import os
import re
import pandas as pd
import numpy as np
from datetime import date, timedelta
import datetime as dtm
import matplotlib.pyplot as plt
import seaborn as sns

# Import DataGrab Functions
from COVID_DataGrab import *

# Import Model Fitting Functions
from collections import *
import torch

# Define COVID Gaussian Process Model Class


class COVID_GP_Model(object):
    def __init__(self, init_param_dct, train_dat_x, train_dat_y,
                 kernel_func, mean_func=lambda t, p: 0,
                 model_name=None):
        # Store Model Values
        self.kernel_func = kernel_func
        self.mean_func = mean_func
        self.model_name = model_name

        # Store training data
        if (train_dat_x.shape[0] - train_dat_y.shape[0]):
            raise BaseException(
                "Error: First dimension size of train_dat_x and train_dat_y must match.")
        else:
            self.train_dat_x = torch.tensor(train_dat_x,
                                            requires_grad=False)
            self.train_dat_y = torch.tensor(train_dat_y,
                                            requires_grad=False)
            self.train_dat_len = train_dat_x.shape[0]

        # Store model parameters
        self.parameters = {}
        for ky in init_param_dct.keys():
            self.parameters[ky] = torch.tensor(init_param_dct[ky],
                                               requires_grad=True,
                                               dtype=torch.float64)

        # Compute log_likelihood for the first time
        self.update_log_likelihood()

    def compute_Kxy(self, x_vec1, x_vec2):
        return self.kernel_func(x_vec1.repeat(x_vec2.shape[0], 1).T,
                                x_vec2.repeat(x_vec1.shape[0], 1),
                                self.parameters)

    def compute_mu(self, x_vec):
        return self.mean_func(x_vec, self.parameters)

    def update_log_likelihood(self):
        # Compute Kernel Matrix, Inverse, etc.
        self.Kxx_train = self.compute_Kxy(self.train_dat_x,
                                          self.train_dat_x)

        self.Kxx_train_inv = self.Kxx_train.inverse()
        self.mu_train = self.compute_mu(self.train_dat_x)

        # Initial Compute of Various Model Params and Likelihood
        self.adj_train_dat_y = (self.train_dat_y - self.mu_train).unsqueeze(-1)
        self.log_likelihood = -(0.5 / self.train_dat_len) * \
            torch.chain_matmul(self.adj_train_dat_y.T,
                               self.Kxx_train_inv,
                               self.adj_train_dat_y)

    def predict(self, x_vec_new, compute_var_mat=False):
        # Compute Posterior Mean
        Kyx = self.compute_Kxy(x_vec_new, self.train_dat_x)
        post_mean = self.compute_mu(x_vec_new) + \
            torch.chain_matmul(Kyx, self.Kxx_train_inv, self.adj_train_dat_y)

        # If requested compute posterior covariance matrix
        if compute_var_mat:
            post_var = self.compute_Kxy(x_vec_new, x_vec_new) - \
                torch.chain_matmul(Kyx, self.Kxx_train_inv, Kyx.T)
        else:
            post_var = None

        return post_mean, post_var

    def __str__(self):
        return "Model Name: " + str(self.model_name) + "\n" + \
            "Current Params: " + str(self.parameters) + "\n" + \
            "Current Loss:  " + str(self.log_likelihood)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':

    # Try to Download Latest Data to Directory
    covid_csv_root = os.path.expanduser(
        '~/Documents/GitHub/COVID-Experiments/CSV/')

    # Attempt to fetch latest data from ECDC
    covid_ecdc_data_download(covid_csv_root)

    # Load Latest CSV file
    covid = covid_ecdc_load_latest(covid_csv_root)

    # Enrich Data
    covid = enrich_covid_dataframe(covid)

    # Subset Full Dataset to Smaller Training Set
    covid_train = (covid
                   [covid.SumCases > 500]
                   [covid.CountryZeroRate < 0.75]
                   [covid.Cases > 0]
                   [['GeoId', 'DaysSinceMin',
                     'Cases', 'lnCasesNorm',
                     'CountryFirstCaseDate', 'CountryPop']])

    # Generate dict of arrays for training
    covid_train_arr = [(covid_train
                        [covid_train.GeoId == c]
                        [['DaysSinceMin', 'lnCasesNorm']]
                        .to_numpy().T) for c in covid_train.GeoId.unique()]

    covid_train_arr = dict(zip(covid_train.GeoId.unique(), covid_train_arr))

    # First Case Dates by Country
    first_case_dates = covid_train.groupby(
        'GeoId').first().CountryFirstCaseDate

    # Country Population
    country_population = covid_train.groupby('GeoId').first().CountryPop

    # Define Kernel and Mean Functions to Use
    def ar_kernel(x_1, x_2, param_dct):
        return param_dct['sigma0']*(param_dct['rho']**torch.abs(x_1-x_2)) + \
            param_dct['sigma1']*(x_1 == x_2)

    def ar_kernel(x_1, x_2, param_dct):
        return param_dct['sigma1']*(x_1 == x_2)

    def quad_mean_function(t_vec, param_dct):
        return (param_dct['beta0'] +
                (param_dct['beta1'] *
                 (t_vec - param_dct['tInit']) *
                 (t_vec - (param_dct['tInit'] + param_dct['tDelta']))))

    # Define Initial Parameters
    init_params = {'rho': 0.0,
                   'sigma0': 1.0, 'sigma1': 1.0,
                   'tInit': 0.0, 'tDelta': 100.0,
                   'beta0': 0.0, 'beta1': 1.0}

    init_params = {'sigma1': 1.0,
                   'tInit': 0.0, 'tDelta': 100.0,
                   'beta0': 0.0, 'beta1': 0.001}

    # Initialize Per-Country Models
    ctry_init_param_dct = {}
    ctry_model_dct = {}

    for ctry_id in covid_train.GeoId.unique():
        # Define initial parameter dictionary for each country
        ctry_init_param_dct[ctry_id] = init_params.copy()
        ctry_init_param_dct[ctry_id].update([('tInit', first_case_dates[ctry_id]),
                                             ('beta0', -1*np.log(country_population[ctry_id]))])

        # Initialize GP model for each country
        ctry_model_dct[ctry_id] = \
            COVID_GP_Model(init_param_dct=ctry_init_param_dct[ctry_id],
                           train_dat_x=covid_train_arr[ctry_id][0, ],
                           train_dat_y=covid_train_arr[ctry_id][1, ],
                           kernel_func=ar_kernel,
                           mean_func=quad_mean_function,
                           model_name=ctry_id)

    # Train those suckers!
    lrn_rate = 1e-5
    n_iter = 1000

    for ctry_id in covid_train.GeoId.unique():  # Train each per-country model
        for i in range(n_iter):  # Run n_iter iterations of gradient descent
            # Update log-likelihood and backprop through
            ctry_model_dct[ctry_id].update_log_likelihood()
            ctry_model_dct[ctry_id].log_likelihood.backward()

            # Print Train Status on Occasion
            if not( (i+1) % 200):
                print("\nIteration %i on Country %s" % (i+1, ctry_id))
                print(ctry_id, "Log-Lik Value:",
                      ctry_model_dct[ctry_id].log_likelihood)
                print(ctry_model_dct[ctry_id].parameters)

            # For each parameter, update using gradient obtained through backprop
            for k in ctry_model_dct[ctry_id].parameters.keys():
                with torch.no_grad():
                    ctry_model_dct[ctry_id].parameters[k] += lrn_rate * \
                        ctry_model_dct[ctry_id].parameters[k].grad

                ctry_model_dct[ctry_id].parameters[k].grad.zero_()
