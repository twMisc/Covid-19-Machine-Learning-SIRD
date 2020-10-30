#%%
from scipy.optimize import minimize
import numpy as np
#port tensorflow as tf 
import os
#import optuna
import matplotlib.pyplot as plt
from load_data import get_all_time_series
from load_data import to_float
from load_data import to_float_vec


#%% 
mypath = r'../COVID-19/'
subpath = r'csse_covid_19_data/csse_covid_19_time_series'
the_path = os.path.join(mypath,subpath)
[df_infected,df_confirmed,df_recovered,df_deaths] = get_all_time_series(the_path)
from information import df_information

countries = ['Italy']
country_code = countries[0]
infected = df_infected[country_code].values
deaths = df_deaths[country_code].values
recovered = df_recovered[country_code].values
population = df_information[df_information.country==country_code].population.values[0]
Regr = int(df_information[df_information.country==country_code].Regr.values[0])
# %% find start_time
start_time = 0
for i in infected:
    if i!=0:
        break
    start_time = start_time+1
print(start_time)
# %% enter end_time
num_times  = 180
end_time  = num_times + start_time 
print(end_time)
# %% Set the initial value for model
i0 = to_float(infected[start_time])
r0 = 0.
s0 = to_float(population) - i0
d0 = to_float(deaths[start_time])
#num_times = len(deaths)
# %%
infected = infected[start_time:end_time]
deaths = deaths[start_time:end_time]
recovered = recovered[start_time:end_time]
infected = to_float_vec(infected)
deaths = to_float_vec(deaths)
recovered = to_float_vec(recovered)
#%%
def sigmoid(x):
    return 1. / (1 + np.exp(-x))
def logit(x):
    return np.log(x / (1-x))
#%%
def minf(x):
    beta = x[0]
    gamma = x[1]
    mu = x[2]

    beta = sigmoid(beta)
    gamma = sigmoid(gamma)
    mu = sigmoid(mu)

    S = []
    I = []
    R = []
    D = []

    S.append(s0)
    I.append(i0)
    R.append(r0)
    D.append(d0)

    for i in range(Regr-1):
        Snew = S[i] - beta/population * S[i]*I[i]
        Inew = I[i] + beta/population * S[i]*I[i] - gamma*I[i] - mu*I[i]
        Rnew = R[i] + gamma* I[i]
        Dnew = D[i] + mu*I[i]
        S.append(Snew)
        I.append(Inew)
        R.append(Rnew)
        D.append(Dnew)
        
    I = np.array(I)
    D = np.array(D)
    
    epsilon = 0.0001
    Ed1 = np.sum((np.log(infected[start_time:Regr+start_time]+epsilon)-np.log(I+epsilon))**2 + (np.log(deaths[start_time:Regr+start_time]+epsilon)-np.log(D+epsilon))**2)
    Ed2 = 0.01*(np.log(np.max(infected[start_time:Regr+start_time])+epsilon)/np.max(infected[start_time:Regr+start_time])) * np.sum((infected[start_time:Regr+start_time]-I)**2+ (deaths[start_time:Regr+start_time]-D)**2)
    #sumr = 0
    #for it in range(num_times-1):
    #    sumr = sumr + (beta[it]-beta[it+1])**2 + (gamma[it]-gamma[it+1])**2 + 100*(mu[it]-mu[it+1])**2
    #Er = 100*np.log(np.max(infected)+epsilon)/0.7215546733084348*sumr
    
    #E0 = 100*np.log(np.max(infected)+epsilon)/0.7215546733084348*((beta[0]-0.7215546733084348)**2+(gamma[0]- 0.5260372903228209)**2 +100*(mu[0]-0.003972672766524191)**2)

    #return np.sum((infected[start_time:Regr+start_time]-I)**2) + 100*np.sum((deaths[start_time:Regr+start_time]-D)**2)
    #print(Ed1+Ed2+Er+E0)
    return Ed1 + Ed2
res = minimize(minf,logit(np.array([0.5,0.5,0.005])),method = 'L-BFGS-B',options={'disp': True,'maxiter':30000,'maxfun':80000})
# %%
beta1 = sigmoid(res.x[0])
gamma1 = sigmoid(res.x[1])
mu1 =sigmoid(res.x[2])

# %%
S = []
I = []
R = []
D = []

S.append(s0)
I.append(i0)
R.append(r0)
D.append(d0)

for i in range(Regr-1):
    Snew = S[i] - beta1/population * S[i]*I[i]
    Inew = I[i] + beta1/population * S[i]*I[i] - gamma1*I[i] - mu1*I[i]
    Rnew = R[i] + gamma1* I[i]
    Dnew = D[i] + mu1*I[i]
    S.append(Snew)
    I.append(Inew)
    R.append(Rnew)
    D.append(Dnew)

I = np.array(I)
D = np.array(D)
R = np.array(R)
S = np.array(S)

# %%
plt.plot(np.arange(len(infected[start_time:Regr+start_time])),infected[start_time:Regr+start_time],label = 'true')
plt.plot(np.arange(len(I)),I , label = 'predicted')
plt.legend()
plt.title('infected')
# %%
plt.plot(np.arange(len(infected[start_time:Regr+start_time])),deaths[start_time:Regr+start_time],label = 'true')
plt.plot(np.arange(len(I)),D , label = 'predicted')
plt.legend()
plt.title('deaths')

# %%
print(beta1)
print(gamma1)
print(mu1)
# %% 
from scipy.optimize import minimize

def minf(x):
    beta = x[0:num_times]
    gamma = x[num_times:2*num_times]
    mu = x[2*num_times:3*num_times]

    beta = sigmoid(beta)
    gamma = sigmoid(gamma)
    mu = sigmoid(mu)

    S = []
    I = []
    R = []
    D = []

    S.append(s0)
    I.append(i0)
    R.append(r0)
    D.append(d0)

    for i in range(num_times-1):
        Snew = S[i] - beta[i]/population * S[i]*I[i]
        Inew = I[i] + beta[i]/population * S[i]*I[i] - gamma[i]*I[i] - mu[i]*I[i]
        Rnew = R[i] + gamma[i]* I[i]
        Dnew = D[i] + mu[i]*I[i]
        S.append(Snew)
        I.append(Inew)
        R.append(Rnew)
        D.append(Dnew)
        
    I = np.array(I)
    D = np.array(D)
    
    epsilon = 0.0001
    Ed1 = np.sum((np.log(infected+epsilon)-np.log(I+epsilon))**2 + (np.log(deaths+epsilon)-np.log(D+epsilon))**2)
    Ed2 = 0.01*(np.log(np.max(infected)+epsilon)/np.max(infected)) * np.sum((infected-I)**2+ (deaths-D)**2)
    sumr = 0
    for it in range(num_times-1):
        sumr = sumr + (beta[it]-beta[it+1])**2 + (gamma[it]-gamma[it+1])**2 + 100*(mu[it]-mu[it+1])**2
    Er = 100*np.log(np.max(infected)+epsilon)/0.7215546733084348*sumr
    
    E0 = 100*np.log(np.max(infected)+epsilon)/0.7215546733084348*((beta[0]-0.7215546733084348)**2+(gamma[0]- 0.5260372903228209)**2 +100*(mu[0]-0.003972672766524191)**2)

    #return np.sum((infected-I)**2) + 100*np.sum((deaths-D)**2)
    #print(Ed1+Ed2+Er+E0)
    return Ed1 + Ed2 + Er + E0

#bnds = [(0, 1) for i in range(end_time*3)]
beta_0 = np.array([beta1 for i in range(num_times)])
gamma_0 = np.array([gamma1 for i in range(num_times)])
mu_0 = np.array([mu1 for i in range(num_times)])
res = minimize(minf,logit(np.array([beta_0,gamma_0,mu_0])),method = 'L-BFGS-B',options={'disp': True,'maxiter':30000,'maxfun':80000})
# %%
beta1 = sigmoid(res.x[0:num_times])
gamma1 = sigmoid(res.x[num_times:2*num_times])
mu1 =sigmoid(res.x[2*num_times:3*num_times])

# %%
S = []
I = []
R = []
D = []

S.append(s0)
I.append(i0)
R.append(r0)
D.append(d0)

for i in range(num_times-1):
    Snew = S[i] - beta1[i]/population * S[i]*I[i]
    Inew = I[i] + beta1[i]/population * S[i]*I[i] - gamma1[i]*I[i] - mu1[i]*I[i]
    Rnew = R[i] + gamma1[i]* I[i]
    Dnew = D[i] + mu1[i]*I[i]
    S.append(Snew)
    I.append(Inew)
    R.append(Rnew)
    D.append(Dnew)

I = np.array(I)
D = np.array(D)
R = np.array(R)
S = np.array(S)

# %%
plt.plot(np.arange(len(infected)),infected,label = 'true')
plt.plot(np.arange(len(I)),I , label = 'predicted')
plt.legend()
plt.title('infected')
# %%
plt.plot(np.arange(len(infected)),deaths,label = 'true')
plt.plot(np.arange(len(I)),D , label = 'predicted')
plt.legend()
plt.title('deaths')

# %%
plt.plot(np.arange(num_times),beta1)
plt.plot(np.arange(num_times),gamma1)

# %%
plt.plot(np.arange(num_times),mu1)
# %%
