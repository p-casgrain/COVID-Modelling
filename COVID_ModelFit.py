# DataGrab Dependencies (not sure how to skip this)
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
import datetime as dtm
from collections import *

# Import DataGrab Functions
from COVID_DataGrab import *

# Import Model Fitting Functions
import torch

# Define COVID Model Class


class COVID_AR_Model1(object):
    def __init__(self, initParamDct, train_dat):
        # Store parameters and training data
        self._parameters = initParamDct.copy()  # No name check
        self._train_dat = torch.tensor(train_dat.copy(),
                                       requires_grad=False)  # No size and type check
        # Initial Compute of Various Model Params and Likelihood
    def _compute_likelihood(self, obs_tensor):
        return None
    def _ar_matrix(self, t_vec):
        return None
    def _quad_mean_function(self, t_vec):
        return None


if __name__ == '__main__':

    # Try to Download Latest Data to Directory
    covid_csv_root = os.path.expanduser('~/Desktop/COVID-Modelling/CSV/')

    # Attempt to fetch latest data
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

    # Generate dict of torch tensors for training
    covid_train_arr = [(covid_train
                        [covid_train.GeoId == c]
                        [['DaysSinceMin', 'lnCasesNorm']]
                        .to_numpy()) for c in covid_train.GeoId.unique()]

    covid_train_arr = [torch.tensor(x, requires_grad=False)
                       for x in covid_train_arr]

    covid_train_arr = dict(zip(covid_train.GeoId.unique(),
                               covid_train_arr))

    # First Case Dates by Country
    first_case_dates = covid_train.groupby(
        'GeoId').first().CountryFirstCaseDate

    # Define Initial Parameters
    init_params = {'rho': 1e-3, 'sigma': 1e-3,
                   'tInit': 0, 'tDelta': 100,
                   'beta0': 0, 'beta1': 1}

    # Initialize Per-Country Models
    ctry_param_dct = {}
    ctry_obj_dct = {}

    for id in covid_train.GeoId.unique():
        # Define country initial parameter dictionary
        ctry_param_dct[id] = init_params.copy()
        ctry_param_dct[id].update([('tInit', first_case_dates[id])])

        # Initialize Country Model Objects
        ctry_obj_dct[id] = COVID_AR_Model1(ctry_param_dct[id],
                                           covid_train_arr[id])

    # Train those suckers!
    for id in covid_train.GeoId.unique():
        
