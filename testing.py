#%% import
import pandas as pd 
import numpy as np 
# %%
import os 
mypath = r'./COVID-19/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv'
df = pd.read_csv(mypath)
# %%
S = set()
for s in df['Country_Region'].values:
    S.add(s)
# %%
len(S)
S = list(S)
S = sorted(S)
# %%
populations = []
for s in S:
    populations.append(df[(df['Country_Region'] == s) & (pd.isna(df['Province_State']))]['Population'].values[0])

# %%
Regrs = np.ones(len(S))*70
iteration = 0
for s in S:
    if s == 'Italy':
        Regrs[iteration]=40
    if s ==  'US':
        Regrs[iteration] = 70
    if s ==  'Germany':
        Regrs[iteration] = 60
    if s ==  'United Kingdom':
        Regrs[iteration] = 60
    if s == 'Spain':
        Regrs[iteration] = 50
    if s == 'France':
        Regrs[iteration] = 60
    if s == 'China':
        Regrs[iteration] = 15
    if s == 'Belgium':
        Regrs[iteration] = 55
    iteration = iteration +1 
# %%
the_dict = {"country": S,  
        "population": populations,
        'Regr' : Regrs
        }
# %%
the_dict['population']

# %%
df_information = pd.DataFrame(the_dict)
# %%
