import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
import datetime as dtm


def get_timedelta_days(dt_series):
    return [x.days for x in dt_series]


def covid_ecdc_data_download(covid_csv_root):
    try:
        # Generate COVID19 Data URL
        yday_str = date.strftime(date.today() - timedelta(1), '%Y-%m-%d')
        url_template = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-%s.csv"
        url_str = url_template % yday_str

        # Download Data from ECE=DC
        file_name = 'COVID-19-' + yday_str + '.csv'
        covid_tmp = pd.read_csv(url_str)
        covid_tmp.to_csv(os.path.join(covid_csv_root, file_name), index=False)

        # Wrap Up
        print("Data for date=%s downloaded from ECDC." % yday_str)
        # del yday_str, url_template, url_str, file_name, covid_tmp

    except:
        print("Latest data could not be downloaded from ECDC.")

    return None


def covid_ecdc_load_latest(covid_csv_root):
    # Import Data from Disk
    latest_filename = list(filter(lambda s: re.match("^\d", s),
                                  os.listdir(covid_csv_root)))
    latest_filename.sort()
    latest_filename = latest_filename[-1]

    covid_tbl = pd.read_csv(
        os.path.join(covid_csv_root, latest_filename),
        squeeze=True,
        parse_dates=False)

    return covid_tbl


def covid_enriched_load_latest(covid_csv_root):
    # Import Data from Disk
    latest_filename = list(filter(lambda s: re.match("^enriched-\d", s),
                                  os.listdir(covid_csv_root)))
    latest_filename.sort()
    latest_filename = latest_filename[-1]

    covid_tbl = pd.read_csv(
        os.path.join(covid_csv_root, latest_filename),
        squeeze=True,
        parse_dates=False)

    return covid_tbl


def enrich_covid_dataframe(covid):

    # Rename Columns for Simplicity
    covid = covid.rename(
        columns={'dateRep': 'DateStr',
                 'countriesAndTerritories': 'Country',
                 'deaths':'Deaths',
                 'cases':'Cases',
                 'geoId':'GeoId',
                 'popData2018': 'CountryPop'})

    # Add Integer Date and Sort
    covid = (covid
             .assign(Date=pd.to_datetime(covid.DateStr, format='%d/%m/%Y'),
                     Country=list(map(lambda x: x.upper(), covid.Country)))
             .assign(DaysSinceMin=lambda x: get_timedelta_days(x.Date - np.min(x.Date)))
             .sort_values(by=['Country', 'Date']))

    # Add Total, Cumulative Cases and Deaths by Country
    covid[['SumCases', 'SumDeaths']] = (covid
                                        .groupby('GeoId')['Cases', 'Deaths']
                                        .transform(np.sum))

    covid[['CumCases', 'CumDeaths']] = (covid
                                        .groupby('GeoId')['Cases', 'Deaths']
                                        .transform(np.cumsum))

    # Find Rate of Zero Cases in Each Country
    covid = covid.join((covid
                        .groupby('GeoId')
                        .apply(lambda x: np.mean(x.Cases == 0))
                        .rename('CountryZeroRate')), on='GeoId')

    # Add Normalized Log-Cases and Log-Deaths
    covid = covid.assign(
        lnCasesNorm=lambda x: np.log(x.Cases/x.CountryPop),
        lnDeathsNorm=lambda x: np.log(x.Deaths/x.CountryPop))

    # Add Date of First Case and First Death
    covid = covid.join((covid
                        .groupby('GeoId')
                        .apply(lambda x: np.min(x.DaysSinceMin[x.Cases > 0]))
                        .rename("CountryFirstCaseDate")),
                       on='GeoId')

    return covid

    # if __name__ == '__main__':

    #     # Try to Download Latest Data to Directory
    #     covid_csv_root = os.path.expanduser('~/Desktop/COVID-Modelling/CSV/')

    #     # Attempt to fetch latest data
    #     covid_ecdc_data_download(covid_csv_root)

    #     # Load Latest CSV file
    #     covid_dt = covid_ecdc_load_latest(covid_csv_root)

    #     # Enrich Data
    #     covid_dt = enrich_covid_dataframe(covid_dt)
    #     covid_dt.groupby('Country').apply(
    #         lambda x: np.min(x.Date[x.Cases > 0])).sort_values()

    #     # Plot the responses for different events and regions
    #     plt.figure(1, figsize=[10, 10])
    #     plt.clf()
    #     sns.set(style="darkgrid")
    #     sns.lineplot(x="Date", y="lnCases",
    #                  hue="Country",
    #                  data=covid[covid.SumCases > 1000][covid.Cases > 0],
    #                  legend='brief')
    #     plt.xlim(covid.Date.min(), covid.Date.max())
    #     plt.show()
