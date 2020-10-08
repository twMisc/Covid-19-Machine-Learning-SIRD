#%% imported need packages
import numpy as np
import tensorflow as tf 
import os
import optuna
import matplotlib.pyplot as plt
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
from information import df_information
print(df_information)

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
        Regr = int(df_information[df_information.country==country_code].Regr.values[0])
    except Exception:
        print('Invalid country code, see country_list.txt for options.')
        continue
    else:
        break

#%% find start_time
start_time = 0
for i in infected:
    if i!=0:
        break
    start_time = start_time+1
print(start_time)
#%% enter end_time
num_times  = int(input('Enter the days to train:'))
end_time  = num_times + start_time 
print(end_time)
# %% Set the initial value for model
i0 = to_float(infected[start_time])
r0 = 0.
s0 = to_float(population) - i0
d0 = to_float(deaths[start_time])
#num_times = len(deaths)
#%%
infected = infected[start_time:end_time]
deaths = deaths[start_time:end_time]
recovered = recovered[start_time:end_time]

#%% Set the training data
x_train = to_float_vec(np.array([infected,deaths]).transpose())
x_trains = to_float_vec(np.array([x_train]))

# %% Solve for the initial parameters
def objective(trial):
    beta = trial.suggest_uniform('beta', 0, 1)
    gamma = trial.suggest_uniform('gamma', 0, 1)
    mu = trial.suggest_uniform('mu', 0, 1)
    
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
    
    #epsilon = 0.000001
    #Ed1 = np.sum((np.log(Ic+epsilon)-np.log(I+epsilon))**2 + (np.log(Dc+epsilon)-np.log(D+epsilon))**2)
    #Ed2 = 0.01*(np.log(np.max(Ic)+epsilon)/np.max(Ic)) * np.sum((Ic-I)**2+ (Dc-D)**2)
    return np.sum((x_train[:,0][0:Regr]-I)**2) + 100*np.sum((x_train[:,1][0:Regr]-D)**2)
#%%
study = optuna.create_study()
study.optimize(objective, n_trials=3000)
study.best_params  # E.g. {'x': 2.002108042}

beta = study.best_params['beta']
gamma = study.best_params['gamma']
mu = study.best_params['mu']

beta_0 = np.array([beta for i in range(num_times)])
gamma_0 = np.array([gamma for i in range(num_times)])
mu_0 = np.array([mu for i in range(num_times)])

# %% Define our custom loss function
@tf.function()
def tf_loss_fn_us(y_true, y_pred):
    beta_t = y_pred[0][:,0]
    gamma_t = y_pred[0][:,1]
    mu_t = y_pred[0][:,2]
    
    beta_0 = y_true[0][0,0]
    gamma_0 = y_true[0][0,1]
    mu_0 = y_true[0][0,2]
    
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
        
    I = tf.stack(I)
    D = tf.stack(D)
    epsilon = 0.000001
    Ed1 = tf.math.reduce_sum((tf.math.log((x_trains[0][:,0]+epsilon))-tf.math.log((I+epsilon)))**2 + (tf.math.log((x_trains[0][:,1]+epsilon))-tf.math.log((D+epsilon)))**2)
    
    Ed2 = 0.01*tf.math.log(tf.math.reduce_max(x_trains[0][:,0])+epsilon)/(tf.math.reduce_max(x_trains[0][:,0])+epsilon) * tf.math.reduce_sum((x_trains[0][:,0]-I)**2+ (x_trains[0][:,1]-D)**2)

    sumr = 0
    for it in range(num_times-1):
        sumr = sumr + (beta_t[it]-beta_t[it+1])**2 + (gamma_t[it]-gamma_t[it+1])**2 + 100*(mu_t[it]-mu_t[it+1])**2
    Er = 100*tf.math.log(tf.math.reduce_max(x_trains[0][:,0])+epsilon)/tf.math.reduce_max([beta_0,gamma_0,mu_0])*sumr
    
    E0 = 100*tf.math.log(tf.math.reduce_max(x_trains[0][:,0])+epsilon)/tf.math.reduce_max([beta_0,gamma_0,mu_0])*((beta_t[0]-beta_0)**2+(gamma_t[0]- gamma_0)**2 +100*(mu_t[0]-mu_0)**2)

    return Ed1+Ed2+Er+E0

# %% set up y_trues
y_true = np.zeros((num_times,3))
y_true[:,0] = beta_0[0:num_times]
y_true[:,1] = gamma_0[0:num_times]
y_true[:,2] = mu_0[0:num_times]
y_trues = np.array([y_true])
# %% Initialize the model and pre-train
tf.keras.backend.set_floatx('float64')

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(num_times, 2)),
    tf.keras.layers.Dense(16,activation='sigmoid'),
    tf.keras.layers.Dense(num_times*3,activation = 'sigmoid'),
    tf.keras.layers.Reshape((num_times, 3))
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.012),
              loss = 'mean_squared_error',
              metrics=['accuracy']
             )

model.fit(x_trains,y_trues,verbose=1,epochs=2500)
# %%  The training process
tf.config.experimental_run_functions_eagerly(False)
def train_step(x_trains, y_trues):
    with tf.GradientTape() as tape:
        logits = model(x_trains, training=True)
    
        # Add asserts to check the shape of the output.
        #tf.debugging.assert_equal(logits.shape, (32, 10))
    
        loss_value = loss_object(y_trues, logits)
    loss_history.append(loss_value)
    grads = tape.gradient(loss_value, model.trainable_variables)
    #print(grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables),experimental_aggregate_gradients=True)

def train(epochs):
    for epoch in range(epochs):
        train_step(x_trains, y_trues)
        print ('Epoch {} finished'.format(epoch))

loss_history = []
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00005)
loss_object = tf_loss_fn_us
train(epochs =10000)
np_loss_history = []
for loss in loss_history:
    np_loss_history.append(loss.numpy())

ax = plt.gca()
plt.plot(np.arange(0,len(np_loss_history),1),np_loss_history)

ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.show()
plt.savefig('loss_fn_'+country_code+'_'+str(num_times))

#%% Save the model
model.save('model_'+country_code+'_'+str(num_times))