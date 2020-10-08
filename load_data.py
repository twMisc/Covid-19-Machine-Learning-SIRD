#%% import needed packages
import pandas as pd
import glob,os
import numpy as np

# %% specify the CCSE github path
#mypath = r'../COVID-19/csse_covid_19_data/csse_covid_19_time_series'

#%% to_float
def to_float(x):
    return float(x)
to_float_vec = np.vectorize(to_float)

# %% define the function to load the .csv files
def get_all_time_series(mypath):
    #path = r'../COVID-19/csse_covid_19_data/csse_covid_19_time_series'
    all_files = glob.glob(os.path.join(mypath, "*.csv"))
    for file in all_files:
        df = pd.read_csv(file)
        #confirmed_all = dict()
        if os.path.basename(file) == 'time_series_covid19_confirmed_global.csv':
            df_confirmed = pd.DataFrame()
            for row in df.values:
                confirmed = np.sum(df[df['Country/Region']==row[1]].values[:,4:],0)
                df_confirmed[row[1]] = confirmed
        if os.path.basename(file) == 'time_series_covid19_deaths_global.csv':
            df_recovered = pd.DataFrame()
            for row in df.values:
                recovered = np.sum(df[df['Country/Region']==row[1]].values[:,4:],0)
                df_recovered[row[1]] = recovered
        if os.path.basename(file) == 'time_series_covid19_recovered_global.csv':
            df_deaths = pd.DataFrame()
            for row in df.values:
                deaths = np.sum(df[df['Country/Region']==row[1]].values[:,4:],0)
                df_deaths[row[1]] = deaths
    df_infected = df_confirmed - df_deaths - df_recovered
    return [df_infected,df_confirmed,df_recovered,df_deaths]