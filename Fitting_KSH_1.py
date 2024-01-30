import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import differential_evolution   
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
import pandas as pd
import math 
from datetime import date
from scipy.integrate import solve_ivp
from itertools import product   
import sys
import os
import re

###########

# Reading data
df = pd.read_csv("C:/Users/khonasoge/OneDrive - Delft University of Technology/Documents/Kailas_PhD/Erlotinib data - Marijn/Code/Tumor_lengths.csv")
df = df.loc[~df['ID'].isin(['A1001','A1003', 'A1015', 'E1034'])] # A1001,A1002,A1015 and E1034 have 4 or fewer data points
df_anyue = pd.read_excel("C:/Users/khonasoge/OneDrive - Delft University of Technology/Documents/Kailas_PhD/Erlotinib data - Marijn/Code/Individual parameters_START_TKI patients_AYin.xlsx") 
df_anyue = df_anyue[["PID","PD_Kg(/day)","PD_Kd(/day)","PD_Km(/day)"]]
df_anyue = df_anyue.rename(columns={"PID": "ID", "PD_Kg(/day)": "kg", "PD_Kd(/day)": "kd", "PD_Km(/day)": "km"})
df_anyue = df_anyue.loc[~df_anyue['ID'].isin(['A1001','A1003', 'A1015', 'E1034'])] # A1001,A1003,A1015 and E1034 have 4 or fewer data points
cols = df_anyue.columns.tolist()
cols[2],cols[3]=cols[3],cols[2] # swap kd and km so anyue's columns match ours
df_anyue = df_anyue[cols]

#############

### population model
def system_ode_exp(time, populations, params):
    Spop = populations[0]
    Rpop = populations[1]
    kg = params[0] #growth rate
    km = params[1] #mutation rate
    kd = params[2] #death rate

    dSpop_dt = Spop * (kg - km - kd)
    dRpop_dt = Rpop * kg + Spop * km
    return [dSpop_dt, dRpop_dt]

# Cost function
def cost_func_exp(params, time, data):
    initial_populations = params[3:]
    tspan = [days[0], days[-1]]
    
    pred_data_rk = solve_ivp(system_ode_exp, tspan, initial_populations, 
                             t_eval = np.ravel(days),  args = tuple([params[:3]]))
    
    # if pred_data_rk.y.shape[1] == np.ravel(days).shape[0]:
    spop = pred_data_rk.y[0]
    rpop = pred_data_rk.y[1]
    sum_pops = spop + rpop
    sumsq_error = np.sum((sum_pops - data)**2)
    msq_error = sumsq_error/len(data)
        # rmse = np.sqrt(msq_error)
    # # else: 
    #     msq_error = 1e30
    #     rmse = 1e30

    return msq_error


unique_IDs = df.ID.unique()

master_dictionary = {} # dictionary to store the optimized parameters for each patient 

# Find optimal parameters for each patient
for id in unique_IDs[1:3]:

    days = df.days[df.ID == id].tolist()
    data = df.sum_tumor_lengths[df.ID == id].tolist()
    print("Working on patient:", id, " Number of data points:", len(days))


    # fitting patient data to equations
    bounds = [(0, np.inf), (1, np.inf), (0, np.inf), (0, data[0]), (0, data[0])] # (kg, km, kd, Spop_0, Rpop_0)
    init_params = [2e-1, 2e-2, 1e-1, data[0]/2, data[0]/2]  
    opt_result_exp = minimize(cost_func_exp, init_params, args=(days, data), bounds = bounds, method="nelder-mead")
    # store the optimal parameters and the cost function value (last entry in list)
    master_dictionary[id] = opt_result_exp.x.tolist() + [opt_result_exp.fun] 
    # store the optimal parameters and the cost function value (last entry in list)


n = len(unique_IDs)
ncols = 3
nrows = np.ceil(n / ncols).astype(int)

fig, axs = plt.subplots(nrows, ncols, figsize=(10, 15), constrained_layout=True)

for index, id in enumerate(unique_IDs[1:3], start=0):
   
    days = df.days[df.ID == id].tolist()
    data = df.sum_tumor_lengths[df.ID == id].tolist()
    tspan = np.linspace(0, days[-1], 300)

    optimal_params_log = master_dictionary[id]
    init_populations_log = opt_result_exp.x[3:]

    pred_data_log = solve_ivp(system_ode_rk45, [0,days[-1]],
                            init_populations_log, dense_output=True,
                            args = tuple([opt_result_exp.x[:3]]))
    # sum_pops = np.sum(pred_data_opt.sol(tspan), axis=0)
    sum_pops_log = np.sum(pred_data_log.sol(tspan), axis=0)
    
    #Using anyue's parameters
    anyue_params = df_anyue[df_anyue.ID == id].values.tolist()[0][1:]
    init_populations_anyue = [data[0], 0]
    pred_data_anyue = solve_ivp(system_ode_rk45, [0,days[-1]],
                                init_populations_anyue, dense_output=True,
                                args = tuple([anyue_params]))
    sum_pops_anyue = np.sum(pred_data_anyue.sol(tspan), axis=0)

    ax = axs.flatten()[(index)%len(unique_IDs)]  

    days = df.days[df.ID == id].tolist()
    data = df.sum_tumor_lengths[df.ID == id].tolist()
    tspan = np.linspace(0, days[-1], 300)

    ax.plot(days, data, 'kx', label='Measured Total') # plot data points

    ax.plot(tspan, pred_data_log.sol(tspan)[0], 'b-', label='Sensitive - Log') # plot logistic model
    ax.plot(tspan, pred_data_log.sol(tspan)[1], 'r-', label='Resistant - Log') 
    ax.plot(tspan, sum_pops_log, 'k-', label='Total - Log')

    ax.plot(tspan, pred_data_anyue.sol(tspan)[0], 'b--', label='Sensitive - Exp')
    ax.plot(tspan, pred_data_anyue.sol(tspan)[1], 'r--', label='Resistant - Exp') # plot anyue's model
    ax.plot(tspan, sum_pops_anyue, 'k--', label='Total - Exp')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.set_title(f'Patient: {id}') 

    # Hide empty subplots
for i in range(index+1, nrows*ncols):
    axs.flatten()[i].axis('off') 

legend_elements = [Line2D([0], [0], marker='x', color='k', label='Measured Total', linestyle='None'),
                   Line2D([0], [0], color='b', label='Sensitive - Log'),
                   Line2D([0], [0], color='r', label='Resistant - Log'),
                   Line2D([0], [0], color='k', linestyle= '-' ,label='Total - Log'),
                   Line2D([0], [0], color='b', linestyle= '--' ,label='Sensitive - Exp'),
                   Line2D([0], [0], color='r', linestyle= ':' ,label='Resistant - Exp'),
                   Line2D([0], [0], color='k', linestyle= '-.' ,label='Total - Exp')]

# Add the legend to the figure
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
fancybox=True, shadow=True, ncol=7)

fig.suptitle('Logistic model sensitive and resistant not fixed')  

plt.show()













# Logistic population model
def system_ode_logistic(time, populations, params):
    Spop = populations[0]
    Rpop = populations[1]
    r = params[0] #growth rate
    kmax = params[1] #carrying capacity
    kd = params[2] #death rate due to medication

    dSpop_dt = Spop * (r * (1- (Spop + Rpop)/kmax) - kd)
    dRpop_dt = Rpop * (r * (1- (Spop + Rpop)/kmax))
    return [dSpop_dt, dRpop_dt]

# Cost function
def cost_func_logistic(params, time, data):
    initial_populations = params[3:]
    tspan = [days[0], days[-1]]
    
    pred_data_log = solve_ivp(system_ode_logistic, tspan, initial_populations, 
                             t_eval = np.ravel(days),  args = tuple([params[:3]]))
    
    if pred_data_log.y.shape[1] == np.ravel(days).shape[0]:
        spop = pred_data_log.y[0]
        rpop = pred_data_log.y[1]
        sum_pops = spop + rpop
        sumsq_error = np.sum((sum_pops - data)**2)
        msq_error = sumsq_error/len(data)
        rmse = np.sqrt(msq_error)
    else: 
        msq_error = 1e30
        rmse = 1e30


    return msq_error

# Gompertizian population model
def system_ode_gompertizian(time, populations, params):
    Spop = populations[0]
    Rpop = populations[1]
    r = params[0] #growth rate
    kmax = params[1] #carrying capacity
    lam = params[2] #sensitivity to medication
    m=1

    dSpop_dt = r *np.log(kmax/(Spop+Rpop)) * (1-lam*m)*Spop 
    dRpop_dt = r * np.log(kmax/(Spop+Rpop))*Rpop

    return [dSpop_dt, dRpop_dt]

# Cost function
def cost_func_gompertizian(params, days, data):
    
    initial_populations = params[3:]
    tspan = [days[0], days[-1]]
    
    pred_data = solve_ivp(system_ode_gompertizian, tspan, initial_populations, 
                             t_eval = np.ravel(days),  args = tuple([params[:3]]))
    
    if pred_data.y.shape[1] == np.ravel(days).shape[0]:
        spop = pred_data.y[0]
        rpop = pred_data.y[1]
        sum_pops = spop + rpop
        sumsq_error = np.sum((sum_pops - data)**2)
        msq_error = sumsq_error/len(data)
        rmse = np.sqrt(msq_error)
    else: 
        msq_error = 1e30
        rmse = 1e30

    return msq_error



init_params_df = pd.DataFrame(columns=['ID', 'r', 'K', 'lambda', 'Spop', 'Rpop'])
opt_params_df = pd.DataFrame(columns=['ID', 'r', 'K', 'lambda', 'Spop', 'Rpop', 'MSE'])
opt_params_clean_df = pd.DataFrame(columns=['ID', 'r', 'K', 'lambda', 'Spop', 'Rpop', 'MSE'])

for id in unique_IDs:
    for row in range(len(df2)):
        col1 = df2.loc[row][0]
        if id in col1:
            init_p = list(eval(col1[10:-1]))
            opt_p = list(eval(df2.loc[row][1]))
            init_params_df.loc[len(init_params_df),'ID'] = id
            init_params_df.loc[len(init_params_df)-1, ['r', 'K', 'lambda', 'Spop', 'Rpop']] = init_p
            opt_params_df.loc[len(opt_params_df),'ID'] = id
            opt_params_df.loc[len(opt_params_df)-1, ['r', 'K', 'lambda', 'Spop', 'Rpop', 'MSE']] = opt_p

    temp = opt_params_df[opt_params_df.ID == id]
    temp['MSE'] = pd.to_numeric(temp['MSE'])
    keepidx = temp.MSE.idxmin()
    opt_params_clean_df.loc[len(opt_params_clean_df),:] = temp.loc[keepidx,:]  