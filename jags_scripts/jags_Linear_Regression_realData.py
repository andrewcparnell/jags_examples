"""
Header--------------------

Fitting a linear regression in JAGS

In this code we generate some data from a simple linear regression model (using real world data) and fit is using jags. 
We then intepret the output.


"""
import pyjags
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import *

"""
Maths -------------------------------------------------------------------

Description of the Bayesian model fitted in this file
Notation:
y_i = repsonse variable for observation t=i,..,N
x_i = explanatory variable for obs i
alpha, beta = intercept and slope parameters to be estimated
sigma = residual standard deviation

Likelihood:
y[i] ~ N(alpha + beta * x[i], sigma^2)
Prior
alpha ~ N(0,100) - vague priors
beta ~ N(0,100)
sigma ~ U(0,10) = tau
"""

# JAGS CODE ----------------------------------------------------
# Jags code to fit the model to the simulated data

code = '''
model {
    for (i in 1:N) {
        y[i] ~ dnorm(alpha + beta * x[i], tau)
    }
    # Priors
    alpha ~ dunif(0, 1e-2)
    beta ~ dnorm(0, 1e-2)
    tau <- 1 / sigma^2
    sigma ~ dunif(0, 10)
}
'''


# Real example ------------------------------------------------------------

# Load in the Church and White global tide gauge data
sea_level = pd.read_csv('https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/church_and_white_global_tide_gauge.csv')
sea_level.head(n=5) # First 5 rows

x = sea_level.year_AD
y = sea_level.sea_level_m
plt.plot(x,y)

# Set up the data
print(sea_level.shape)# Check the number of rows & columns
N = 130 # Number of rows  

model = pyjags.Model(code, data=dict(x = x, y = y, N = N), chains = 4) 

# Number of iterations to remove at start and amount of thinning
model.sample(200, vars=[], thin = 2)

# Choose the parameters to watch and iterations:
samples = model.sample(1000, vars=['alpha', 'beta', 'sigma'])

"""
 results ----------------------------------------------------------------
"""
def summary(samples, varname, p=95):
    values = samples[varname]
    ci = np.percentile(values, [100-p, p])
    print('{:<6} mean = {:>5.1f}, {}% credible interval [{:>4.1f} {:>4.1f}]'.format(
      varname, np.mean(values), p, *ci))

for varname in ['alpha', 'beta', 'sigma']:
    summary(samples, varname)
    
    # Use pandas three dimensional Panel to represent the trace:
trace = pd.Panel({k: v.squeeze(0) for k, v in samples.items()})
trace.axes[0].name = 'Variable'
trace.axes[1].name = 'Iteration'
trace.axes[2].name = 'Chain'
 
# Point estimates:
print("Mean:")
print(trace.to_frame().mean()) # .to_frame converts a series to dataframe
 

# Bayesian equal-tailed 95% credible intervals:
print("Credible Intervals:")
print(trace.to_frame().quantile([0.05, 0.95]))
 

def plot(trace, var):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle(var, fontsize='xx-large')
 
    # Marginal posterior density estimate:
    trace[var].plot.density(ax=axes[0])
    axes[0].set_xlabel('Parameter value')
    axes[0].locator_params(tight=True)
 
    # Autocorrelation for each chain:
    axes[1].set_xlim(0, 100)
    for chain in trace[var].columns:
        autocorrelation_plot(trace[var,:,chain], axes[1], label=chain)
 
    # Trace plot:
    axes[2].set_ylabel('Parameter value')
    trace[var].plot(ax=axes[2])
 
   
 # Display diagnostic plots
for var in trace:
    plot(trace, var)
    
    #The means below are the ones obtained from running pyjags:
alpha_mean = 0.004634
beta_mean = -0.000038


#However, something is not quite right here... the means should be closer to the values below (obtained from R2jags):
alpha_mean_R = -3.060865
beta_mean_R = 0.001537255

# Creating a plot:
plt.plot(x,y)
plt.ylabel('sea_level_m')
plt.xlabel('Year')
plt.plot(x, alpha_mean + beta_mean * x, c = "red")
plt.plot(x, alpha_mean_R + beta_mean_R * x, c = "yellow")

