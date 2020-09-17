#%% imported need packages
import numpy as np
import tensorflow as tf 
import os
from load_data import get_all_time_series

# %% load the data from user input
print('Enter the path of CCSE COVID-19 repository:')
print('(Default: previous folder of this file, press ENTER)')

while(True):
    try:
        mypath = input()
        if (mypath == ''):
            if os.name != 'nt':
                mypath = r'../COVID-19/csse_covid_19_data/csse_covid_19_time_series'
            else:
                mypath = r'..\\COVID-19\\csse_covid_19_data\\csse_covid_19_time_series'
        [df_infected,df_confirmed,df_recovered,df_deaths] = get_all_time_series(mypath)
    except Exception:
        print('Path error, please re-enter:')
        continue
    else:
        break

#%% create df of needed information for each country
import pandas as pd
countries = ["US","Taiwan*"]
populations = [327200000,23780000]
Regrs = [70,100]
dict = {"country": countries,  
        "population": populations,
        'Regr':Regrs
        }
df_information = pd.DataFrame(dict)

# %% user input the country to train
while(True):
    try:
        country_code = input('Enter the country to train(Default=US):')
        if (country_code == ''):
            country_code = 'US'
        infected = df_infected[country_code].values
        deaths = df_deaths[country_code].values
        recovered = df_recovered[country_code].values
        population = df_information[df_information.country==country_code].population.values[0]
        Regr = df_information[df_information.country==country_code].Regr.values[0]
    except Exception:
        print('Invalid country code, see country_list.txt for options.')
        continue
    else:
        break
# %%
print('yeah')

# %%
