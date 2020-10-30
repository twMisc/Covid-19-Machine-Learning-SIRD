#%%
import xgboost as xgb
from load_data import get_all_time_series
from load_data import to_float_vec
import os
#%%
mypath = r'../COVID-19/'
subpath = r'csse_covid_19_data/csse_covid_19_time_series'
the_path = os.path.join(mypath,subpath)
[df_infected,df_confirmed,df_recovered,df_deaths] = get_all_time_series(the_path)
#%% create df of needed information for each country
from information import df_information
#print(df_information)
# %% user input the country to train
countries = ['US','Spain','Belgium','China','France','Germany','United Kingdom','Italy']
country_code = countries[0]
infected = to_float_vec(df_infected[country_code].values)
deaths = to_float_vec(df_deaths[country_code].values)
recovered = to_float_vec(df_recovered[country_code].values)
population = df_information[df_information.country==country_code].population.values[0]
Regr = int(df_information[df_information.country==country_code].Regr.values[0])

# %%
