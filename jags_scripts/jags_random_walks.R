# Header ------------------------------------------------------------------

# Random walk models (first and second difference)
# Andrew Parnell

# In this code we fit some random walk type models to data

# Some boiler plate code to clear the workspace and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y(t) = response variable at time t, t = 1,...,T
# mu: Optional drift parameter
# sigma = residual standard deviation

# Likelihood:
# Order 1: y(t) - y(t-1) ~ N(mu,sigma^2)
# Order 2: y(t) - 2y(t-1) + y(t-2) ~ N(mu,sigma^2)
# Prior:
# sigma ~ unif(0,100) - vague
# sigma ~ dgamma(a,b) ~ informative with good values for a and b
# mu ~ dnorm(0,100) - vague

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
set.seed(123)
T = 100
sigma = 1
mu = 0
t = 1:T
y = cumsum(rnorm(T,mu,sigma))
y2 = cumsum(cumsum(rnorm(T,mu,sigma)))
plot(t,y)
plot(t,y2)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
# Note: running the differencing offline here as part of the data step
model_code = '
model
{
  # Likelihood
  for (t in 1:N_T) {
    z[t] ~ dnorm(mu, tau)
  }

  # Priors
  mu ~ dnorm(0.0,0.01)
  tau <- 1/pow(sigma,2) # Turn precision into standard deviation
  sigma ~ dunif(0.0,100.0)
}
'

# Set up the data
order = 1
model_data = list(z = diff(y, differences=order), N_T = T-order)

# Choose the parameters to watch
model_parameters =  c("mu","sigma")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code),
                 n.chains=4, # Number of different starting positions
                 n.iter=1000, # Number of iterations
                 n.burnin=200, # Number of iterations to remove at start
                 n.thin=2) # Amount of thinning

# Try the order 2 version
order = 2
model_data_2 = list(z = diff(y2, differences=order), N_T = T-order)

model_run_2 = jags(data = model_data_2,
                   parameters.to.save = model_parameters,
                   model.file=textConnection(model_code),
                   n.chains=4, # Number of different starting positions
                   n.iter=1000, # Number of iterations
                   n.burnin=200, # Number of iterations to remove at start
                   n.thin=2) # Amount of thinning

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
print(model_run)
print(model_run_2)

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory
ice = read.csv('https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/GISP2_20yr.csv')
head(ice)
with(ice,plot(Age,Del.18O,type='l'))
# Try plots of differences
with(ice,plot(Age[-1],diff(Del.18O,differences=1),type='l'))
with(ice,plot(Age[-(1:2)],diff(Del.18O,differences=2),type='l'))

# Have to be careful here with differences, look:
table(diff(ice$Age))
# The random walk model above requires evenly spaced time differences

# Just use the Holocene as most of this is 20-year
ice2 = subset(ice,Age<=10000)
table(diff(ice2$Age))

# Look at acf and pacf
acf(ice2$Del.18O)
pacf(ice2$Del.18O)

# Set up the data
order = 1
real_data = with(ice2,
                 list(z = diff(Del.18O, differences=order), N_T = T-order))

# Run the model
real_data_run = jags(data = real_data,
                     parameters.to.save = model_parameters,
                     model.file=textConnection(model_code),
                     n.chains=4,
                     n.iter=1000,
                     n.burnin=200,
                     n.thin=2)

print(real_data_run)

# Create some predictions based on the fitted values
post = print(real_data_run)

# Create some predictions off into the future
T_future = -200 # Remember we're in years before 1950 here so -50 = 2000, also on 20 year grid
future_grid = seq(min(ice2$Age), T_future, by=-20)
future_values = matrix(NA, ncol=length(future_grid), nrow=1000) # Create 1000 future simulations
for(i in 1:1000) {
  future_values[i,1] = ice2$Del.18O[1] + rnorm(1,
                                               mean = real_data_run$BUGSoutput$sims.list$mu[i],
                                               sd = real_data_run$BUGSoutput$sims.list$sigma[i])
  for(j in 2:length(future_grid)) {
    future_values[i,j] = future_values[i,j-1] + rnorm(1,
                                                 mean = real_data_run$BUGSoutput$sims.list$mu[i],
                                                 sd = real_data_run$BUGSoutput$sims.list$sigma[i])
  }
}

# Summarise them
future_low = apply(future_values,2,'quantile',0.25)
future_high = apply(future_values,2,'quantile',0.75)
future_med = apply(future_values,2,'quantile',0.5)

# Summarise Plot these all together
with(ice2,
     plot(Age,
          Del.18O,
          type = 'l',
          xlim = range(c(Age,future_grid)),
          ylim = range(c(Del.18O,future_values))))
lines(future_grid,future_low,lty='dotted',col='red')
lines(future_grid,future_high,lty='dotted',col='red')
lines(future_grid,future_med,col='red')


# Clearly a lot of uncertainty!

# Other tasks -------------------------------------------------------------

# 1) Try running the ice core data with the second order RW. Is there a big difference in the parameter values?
# 2) Try running the ice core data on an older period of the data, e.g. 20k to 80k years ago (check the time differences are roughly equal)
# 3) (Harder) Go back to the sea level data set used in the linear regression example. See if you can include a random walk term together with the linear regression terms. Does it improve the model? (Hint: look at the residuals)

