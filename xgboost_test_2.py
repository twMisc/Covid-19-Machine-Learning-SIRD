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
x_test = []
y_test = []
x_train = []
y_train = []
# %% create the dataset 
S = df_confirmed.keys()
import random
test_size = 0.33
S_test = random.sample(list(S),int(test_size*len(S)))
S_train = list(set(S) - set(S_test))

for s in S_test:
    country_code = s    
    infected = to_float_vec(df_infected[country_code].values)
    deaths = to_float_vec(df_deaths[country_code].values)
    recovered = to_float_vec(df_recovered[country_code].values)
    population = df_information[df_information.country==country_code].population.values[0]
    #%% set time vairables
    num_times = len(infected)
    # %% generate
    observe_days = 30
    predict_days = 1
    days = observe_days + predict_days

    for i in range(num_times - days +1):
        x_test.append(np.concatenate([infected[i:observe_days+i],deaths[i:observe_days+i],]))
        y_test.append(np.concatenate([infected[observe_days+i:days+i],deaths[observe_days+i:days+i]]))

for s in S_train:
    country_code = s    
    infected = to_float_vec(df_infected[country_code].values)
    deaths = to_float_vec(df_deaths[country_code].values)
    recovered = to_float_vec(df_recovered[country_code].values)
    population = df_information[df_information.country==country_code].population.values[0]
    #%% set time vairables
    num_times = len(infected)
    # %% generate
    observe_days = 30
    predict_days = 1
    days = observe_days + predict_days

    for i in range(num_times - days +1):
        x_train.append(np.concatenate([infected[i:observe_days+i],deaths[i:observe_days+i],]))
        y_train.append(np.concatenate([infected[observe_days+i:days+i],deaths[observe_days+i:days+i]]))

# %% split the datset in to train annd test
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=87)
# %% load dataset in xgboost format
infected_regressor = xgb.XGBRegressor().fit(x_train, y_train[:,0])
deaths_regressor = xgb.XGBRegressor().fit(x_train, y_train[:,1])
# %%
infected_regressor.score(x_train,y_train[:,0])
# %%
deaths_regressor.score(x_train,y_train[:,1])
# %%
infected_regressor.score(x_test,y_test[:,0])
# %%
deaths_regressor.score(x_test,y_test[:,1])
#%%
test_target = 13000
#multioutputregressor.predict(x_test)[test_target]
# %%
import matplotlib.pyplot as plt
plt.figure()
plt.title('infected')
plt.plot(np.arange(observe_days),x_test[test_target][0:observe_days],label='observe')
plt.plot(np.arange(predict_days)+observe_days,y_test[test_target][0:predict_days],'-o',label='true')
plt.plot(np.arange(predict_days)+observe_days,infected_regressor.predict(x_test)[test_target],'-o',label='predict')
plt.legend()
# %%
plt.figure()
plt.title("deaths")
plt.plot(np.arange(observe_days),x_test[test_target][observe_days:],label='observe')
plt.plot(np.arange(predict_days)+observe_days,y_test[test_target][predict_days:],'-o',label='true')
plt.plot(np.arange(predict_days)+observe_days,deaths_regressor.predict(x_test)[test_target],'-o',label='predict')
plt.legend()

#%%
for i in range(5):
    country_code = S_test[i]
    infected = to_float_vec(df_infected[country_code].values)
    deaths = to_float_vec(df_deaths[country_code].values)
    recovered = to_float_vec(df_recovered[country_code].values)
    population = df_information[df_information.country==country_code].population.values[0]

    num_times = len(infected)
    observe_days = 30
    predict_days = 1
    days = observe_days + predict_days

    x = []
    #y = []
    for i in range(num_times - days +1):
        x.append(np.concatenate([infected[i:observe_days+i],deaths[i:observe_days+i],]))
        #y.append(np.concatenate([infected[observe_days+i:days+i],deaths[observe_days+i:days+i]]))
    x = np.array(x)
    #y = np.array(y)

    infected_p = infected_regressor.predict(x)
    deaths_p = deaths_regressor.predict(x)
    plt.figure()
    plt.figure()
    plt.title(country_code)
    plt.plot(np.arange(len(infected)),infected,label='true infected')
    plt.plot(np.arange(len(infected_p))+(len(infected)-len(infected_p) +1 ),infected_p,marker = 'x',markersize = '3',linewidth = 0.5,label='predict infected')
    plt.legend()

    plt.figure()
    plt.title(country_code)
    plt.plot(np.arange(len(infected)),deaths,label='true deaths')
    plt.plot(np.arange(len(infected_p))+(len(infected)-len(infected_p) +1 ),deaths_p,marker = 'x',markersize = '3',linewidth = 0.5,label='predict deaths')
    plt.legend()
# %%
