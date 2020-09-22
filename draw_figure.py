# %% load needed packages
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import os
from load_data import get_all_time_series
from load_data import to_float
from load_data import to_float_vec
# %% load the data from user input
print('Enter the path of CCSE COVID-19 repository:')
print('(Default: previous folder of this file, press ENTER)')

while(True):
    try:
        mypath = input()
        if (mypath == ''):
            if os.name != 'nt':
                mypath = r'../COVID-19/'
            else:
                mypath = r'..\\COVID-19\\csse_covid_19_data\\csse_covid_19_time_series'
        subpath = r'csse_covid_19_data/csse_covid_19_time_series'
        the_path = os.path.join(mypath,subpath)
        [df_infected,df_confirmed,df_recovered,df_deaths] = get_all_time_series(the_path)
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
# %% Set the initial value for model
i0 = to_float(infected[0])
r0 = 0.
s0 = to_float(population) - i0
d0 = to_float(deaths[0])

# %% load the trained model
while(True):
    try:
        mypath = input('Enter the folder of the trained model:')
        model= tf.keras.models.load_model(mypath)
    except Exception:
        print('Path error, please re-enter:')\

        continue
    else:
        break
# %% find the model num_times
layer = model.layers[-1]
num_times = layer.output_shape[1]

#%% Set the training data
x_train = to_float_vec(np.array([infected[0:num_times],deaths[0:num_times]]).transpose())
x_trains = to_float_vec(np.array([x_train]))
# %% define the SIRD solver
def solve_SIRD_discrete(num_times,beta_t,gamma_t,mu_t,s0,i0,r0,d0):
    S = []
    I = []
    R = []
    D = []

    S.append(s0)
    I.append(i0)
    R.append(r0)
    D.append(d0)
    num_times = len(beta_t)

    for i in range(num_times-1):
        Snew = S[i] - beta_t[i]/population * S[i]*I[i]
        Inew = I[i] + beta_t[i]/population * S[i]*I[i] - gamma_t[i]*I[i] - mu_t[i]*I[i]
        Rnew = R[i] + gamma_t[i]* I[i]
        Dnew = D[i] + mu_t[i]* I[i]
        S.append(Snew)
        I.append(Inew)
        R.append(Rnew)
        D.append(Dnew)
    return [S,I,R,D]

# %% solve sird with trained model
[Sr,Ir,Rr,Dr]= solve_SIRD_discrete(num_times,model.predict(x_trains)[0][:,0],model.predict(x_trains)[0][:,1],model.predict(x_trains)[0][:,2],s0,i0,r0,d0)


# %% infected figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(num_times),infected/population,marker = 'x',markersize = '3',linewidth = 0.5,label='data')
ax.plot(np.arange(num_times),Ir/population, label = 'model')
ax.legend()
#plt.yscale('log')
ax.set_xlabel('Time [days]')
ax.set_ylabel('I/N [%]')
ax.set_title('Infected')
fig.add_axes(ax)
fig.savefig(r'./img/' + mypath +'/infected.png')

# %% draw infected difference figure 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(num_times-1),np.diff(infected)/population,marker = 'x',markersize = '3',linewidth = 0.5,label='data')
ax.plot(np.arange(num_times-1),np.diff(Ir)/population, label = 'model')
ax.legend()
#plt.yscale('log')
ax.set_xlabel('Time [days]')
ax.set_ylabel('ΔI/N [%/day]')
fig.add_axes(ax)
fig.savefig(r'./img/' + mypath +'/infected_diff.png')

# %% deaths figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(num_times),deaths/population,marker = 'x',markersize = '3',linewidth = 0.5,label='data')
ax.plot(np.arange(num_times),Dr/population, label = 'model')
plt.legend()
#plt.yscale('log')
ax.set_xlabel('Time [days]')
ax.set_ylabel('D/N [%]')
ax.set_title('Deaths')
fig.add_axes(ax)
fig.savefig(r'./img/' + mypath +'/deaths.png')


# %% draw infected difference figure 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(num_times-1),np.diff(deaths)/population,marker = 'x',markersize = '3',linewidth = 0.5,label='data')
ax.plot(np.arange(num_times-1),np.diff(Dr)/population, label = 'model')
plt.legend()
#plt.yscale('log')
ax.set_xlabel('Time [days]')
ax.set_ylabel('ΔD/N [%/day]')
ax.set_title('Deaths')
fig.add_axes(ax)
fig.savefig(r'./img/' + mypath +'/deaths_diff.png')


# %% recovery figure
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(np.arange(num_times),recovered/population,marker = 'x',markersize = '3',linewidth = 0.5,label='data')
ax.plot(np.arange(num_times),Rr/population, label = 'model')
plt.legend()
#plt.yscale('log')
ax.set_xlabel('Time [days]')
ax.set_ylabel('R/N [%]')
ax.set_title('Recovered')
fig.add_axes(ax)
fig.savefig(r'./img/' + mypath +'/recovered.png')

# %% draw recovered difference figure 
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(np.arange(num_times-1),np.diff(recovered)/population,marker = 'x',markersize = '3',linewidth = 0.5,label='data')
ax.plot(np.arange(num_times-1),np.diff(Rr)/population, label = 'model')
plt.legend()
#plt.yscale('log')
ax.set_xlabel('Time [days]')
ax.set_ylabel('ΔR/N [%/day]')
ax.set_title('Recovered')
fig.add_axes(ax)
fig.savefig(r'./img/' + mypath +'/recovered_diff.png')


# %% Suspected figure
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(np.arange(num_times),recovered/population,marker = 'x',markersize = '3',linewidth = 0.5,label='data')
ax.plot(np.arange(num_times),Sr/population, label = 'model')
plt.legend()
#plt.yscale('log')
ax.set_xlabel('Time [days]')
ax.set_ylabel('S/N [%]')
ax.set_title('Suspected')
fig.add_axes(ax)
fig.savefig(r'./img/' + mypath +'/suspected.png')


# %% draw suspected difference figure 
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(np.arange(num_times-1),np.diff(recovered)/population,marker = 'x',markersize = '3',linewidth = 0.5,label='data')
ax.plot(np.arange(num_times-1),np.diff(Sr)/population, label = 'model')
plt.legend()
#plt.yscale('log')
ax.set_xlabel('Time [days]')
ax.set_ylabel('ΔS/N [%/day]')
ax.set_title('Suspected')
fig.add_axes(ax)
fig.savefig(r'./img/' + mypath +'/suspected_diff.png')

# %% draw beta gamma mu
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(num_times),model.predict(x_trains)[0][:,0],label = 'beta')
ax.plot(np.arange(num_times),model.predict(x_trains)[0][:,1],label = 'gamma')
plt.legend()

# %%
