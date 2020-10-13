#%% import 
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import os
from load_data import get_all_time_series
from load_data import to_float
from load_data import to_float_vec

#%% load all data
mypath = r'../COVID-19/'
subpath = r'csse_covid_19_data/csse_covid_19_time_series'
the_path = os.path.join(mypath,subpath)
[df_infected,df_confirmed,df_recovered,df_deaths] = get_all_time_series(the_path)
from information import df_information
# %% load the trained model
betas = dict()
gammas = dict()
mus = dict()
countries = ['US','Spain','Belgium','China','France','Germany','United Kingdom','Italy']
for country_code in countries:
    mypath = 'model_' + country_code + '_180'
    model= tf.keras.models.load_model(mypath)

    infected = df_infected[country_code].values
    deaths = df_deaths[country_code].values
    recovered = df_recovered[country_code].values
    population = df_information[df_information.country==country_code].population.values[0]
    Regr = int(df_information[df_information.country==country_code].Regr.values[0])

    #%% find start_time
    start_time = 0
    for i in infected:
        if i!=0:
            break
        start_time = start_time+1

    # %% Set the initial value for model
    i0 = to_float(infected[start_time])
    r0 = 0.
    s0 = to_float(population) - i0
    d0 = to_float(deaths[start_time])

    # %% find the model num_times
    layer = model.layers[-1]
    num_times = layer.output_shape[1]
    end_time = start_time+num_times
    
    #%% Set the training data
    x_train = to_float_vec(np.array([infected[start_time:end_time],deaths[start_time:end_time]]).transpose())
    x_trains = to_float_vec(np.array([x_train]))

    #%% save betas gammas
    beta = model.predict(x_trains)[0][:,0]
    gamma = model.predict(x_trains)[0][:,1]
    mu = model.predict(x_trains)[0][:,2]
    betas[country_code] = beta
    gammas[country_code] = gamma
    mus[country_code] = mu
# %% beta_gamma plotly

import plotly.graph_objects as go

fig = go.Figure()

# Add traces

fig.add_trace(go.Scatter(x=betas['Italy'], y=gammas['Italy'], mode='markers',name='Italy'))
fig.add_trace(go.Scatter(x=betas['Germany'], y=gammas['Germany'], mode='markers',name='Germany'))
fig.add_trace(go.Scatter(x=betas['United Kingdom'], y=gammas['United Kingdom'], mode='markers',name='UK'))
fig.add_trace(go.Scatter(x=betas['Spain'], y=gammas['Spain'], mode='markers',name='Spain'))
fig.add_trace(go.Scatter(x=betas['US'], y=gammas['US'], mode='markers',name='US'))
fig.add_trace(go.Scatter(x=betas['France'], y=gammas['France'], mode='markers',name='France'))
fig.add_trace(go.Scatter(x=betas['China'], y=gammas['China'], mode='markers',name='China'))
fig.add_trace(go.Scatter(x=betas['Belgium'], y=gammas['Belgium'], mode='markers',name='Belgium'))


fig.update_layout(
    #title="Plot Title",
    xaxis_title="beta",
    yaxis_title="gamma",
)

fig.write_html('./html/beta_gamma_compare.html', auto_open=True)
# %% beta plotly
# Add traces
fig = go.Figure()

fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['Italy'], mode='lines+markers',name='Italy'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['Germany'], mode='lines+markers',name='Germany'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['United Kingdom'], mode='lines+markers',name='UK'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['Spain'], mode='lines+markers',name='Spain'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['US'], mode='lines+markers',name='US'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['France'], mode='lines+markers',name='France'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['China'], mode='lines+markers',name='China'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['Belgium'], mode='lines+markers',name='Belgium'))


fig.update_layout(
    #title="Plot Title",
    yaxis_title="beta",
    xaxis_title="Time [days]",
)

fig.write_html('./html/beta_compare.html', auto_open=True)
# %% gamma plotly
# Add traces
fig = go.Figure()

fig.add_trace(go.Scatter(x=np.arange(num_times), y=gammas['Italy'], mode='lines+markers',name='Italy'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=gammas['Germany'], mode='lines+markers',name='Germany'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=gammas['United Kingdom'], mode='lines+markers',name='UK'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=gammas['Spain'], mode='lines+markers',name='Spain'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=gammas['US'], mode='lines+markers',name='US'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=gammas['France'], mode='lines+markers',name='France'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=gammas['China'], mode='lines+markers',name='China'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=gammas['Belgium'], mode='lines+markers',name='Belgium'))


fig.update_layout(
    #title="Plot Title",
    yaxis_title="gamma",
    xaxis_title="Time [days]",
)

fig.write_html('./html/gamma_compare.html', auto_open=True)

# %% mu plotly
# Add traces
fig = go.Figure()

fig.add_trace(go.Scatter(x=np.arange(num_times), y=mus['Italy'], mode='lines+markers',name='Italy'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=mus['Germany'], mode='lines+markers',name='Germany'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=mus['United Kingdom'], mode='lines+markers',name='UK'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=mus['Spain'], mode='lines+markers',name='Spain'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=mus['US'], mode='lines+markers',name='US'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=mus['France'], mode='lines+markers',name='France'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=mus['China'], mode='lines+markers',name='China'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=mus['Belgium'], mode='lines+markers',name='Belgium'))


fig.update_layout(
    #title="Plot Title",
    yaxis_title="mu",
    xaxis_title="Time [days]",
)

fig.write_html('./html/mu_compare.html', auto_open=True)

#%% mu_beta plotly
fig = go.Figure()

# Add traces

fig.add_trace(go.Scatter(x=mus['Italy'], y=betas['Italy'], mode='markers',name='Italy'))
fig.add_trace(go.Scatter(x=mus['Germany'], y=betas['Germany'], mode='markers',name='Germany'))
fig.add_trace(go.Scatter(x=mus['United Kingdom'], y=betas['United Kingdom'], mode='markers',name='UK'))
fig.add_trace(go.Scatter(x=mus['Spain'], y=betas['Spain'], mode='markers',name='Spain'))
fig.add_trace(go.Scatter(x=mus['US'], y=betas['US'], mode='markers',name='US'))
fig.add_trace(go.Scatter(x=mus['France'], y=betas['France'], mode='markers',name='France'))
fig.add_trace(go.Scatter(x=mus['China'], y=betas['China'], mode='markers',name='China'))
fig.add_trace(go.Scatter(x=mus['Belgium'], y=betas['Belgium'], mode='markers',name='Belgium'))


fig.update_layout(
    #title="Plot Title",
    xaxis_title="mu",
    yaxis_title="beta",
)

fig.write_html('./html/mu_beta_compare.html', auto_open=True)

#%% mu+gamma plotly
fig = go.Figure()

# Add traces

fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['Italy']+gammas['Italy'], mode='lines+markers',name='Italy'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['Germany']+gammas['Germany'], mode='lines+markers',name='Germany'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['United Kingdom']+gammas['United Kingdom'], mode='lines+markers',name='UK'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['Spain']+gammas['Spain'], mode='lines+markers',name='Spain'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['US']+gammas['US'], mode='lines+markers',name='US'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['France']+gammas['France'], mode='lines+markers',name='France'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['China']+gammas['China'], mode='lines+markers',name='China'))
fig.add_trace(go.Scatter(x=np.arange(num_times), y=betas['Belgium']+gammas['Belgium'], mode='lines+markers',name='Belgium'))


fig.update_layout(
    #title="Plot Title",
    xaxis_title="Time [days]",
    yaxis_title="beta+gamma",
)

fig.write_html('./html/beta+gamma_compare.html', auto_open=True)
