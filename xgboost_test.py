#%% load packages
import xgboost as xgb
from load_data import get_all_time_series
from load_data import to_float_vec
import os
#%% load datset from ccse github using written functino
mypath = r'../COVID-19/'
subpath = r'csse_covid_19_data/csse_covid_19_time_series'
the_path = os.path.join(mypath,subpath)
[df_infected,df_confirmed,df_recovered,df_deaths] = get_all_time_series(the_path)
#%% create df of needed information for each country
from information import df_information
#print(df_information)
#%% initialize a matrix for taining sets
import numpy as np 
x_data = []
y_data = []
# %% create the dataset 
S = df_confirmed.keys()
for s in S:
    country_code = s    
    infected = to_float_vec(df_infected[country_code].values)
    deaths = to_float_vec(df_deaths[country_code].values)
    recovered = to_float_vec(df_recovered[country_code].values)
    population = df_information[df_information.country==country_code].population.values[0]
    #%% set time vairables
    num_times = len(infected)
    # %% generate
    observe_days = 50
    predict_days = 3
    days = observe_days + predict_days

    for i in range(num_times - days +1):
        x_data.append(np.concatenate([infected[i:observe_days+i],deaths[i:observe_days+i],]))
        y_data.append(np.concatenate([infected[observe_days+i:days+i],deaths[observe_days+i:days+i]]))
# %% split the datset in to train annd test
x_data = np.array(x_data)
y_data = np.array(y_data)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=87)
# %% load dataset in xgboost format
from sklearn.multioutput import MultiOutputRegressor
multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor()).fit(x_train, y_train)

# %%
multioutputregressor.score(x_train,y_train)
# %%
multioutputregressor.score(x_test,y_test)
# %%
test_target = 11110
#multioutputregressor.predict(x_test)[test_target]
# %%
import matplotlib.pyplot as plt
plt.plot(np.arange(observe_days),x_test[test_target][0:observe_days])
plt.plot(np.arange(predict_days)+observe_days,y_test[test_target][0:predict_days])
plt.plot(np.arange(predict_days)+observe_days,multioutputregressor.predict(x_test)[test_target][0:predict_days])

# %%
plt.plot(np.arange(observe_days),x_test[test_target][observe_days:])
plt.plot(np.arange(predict_days)+observe_days,y_test[test_target][predict_days:])
plt.plot(np.arange(predict_days)+observe_days,multioutputregressor.predict(x_test)[test_target][predict_days:])


#%%
from sklearn.ensemble import RandomForestRegressor
regr_multirf = MultiOutputRegressor(RandomForestRegressor()).fit(x_train,y_train)
# %%
regr_multirf.score(x_train,y_train)

# %%
regr_multirf.score(x_test,y_test)