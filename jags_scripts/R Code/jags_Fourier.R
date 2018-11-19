# Header ------------------------------------------------------------------

# A simple Bayesian Fourier model to produce a periodogram
# Andrew Parnell

# This model creates a periodogram of the data and applies to the Lynx data set example

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_t = Response variable at time t, t=1,...,T
# alpha = Overall mean parameters
# beta = cosine associated frequency coefficient
# gamma = sine associated frequency coefficient
# f_k = frequency value k, for k=1,...,K
# sigma = residual standard deviation

# Likelihood:
# y_t ~ N( mu_t, sigma^2)
# mu_t = beta * cos ( 2 * pi * t * f_k) + gamma * sin ( 2 * pi * t * f_k )
# K and f_k are data and are set in advance
# We fit this model repeatedly (it's very fast) for lots of different f_k

# Priors - all vague here
# alpha ~ normal(0, 100)
# beta ~ normal(0, 100)
# gamma ~ normal(0, 100)
# sigma ~ uniform(0, 100)

# Output quantity:
# We will create the power as:
# P(f_k) = ( beta^2 + gamma^2 ) / 2
# This is what we create for our periodogram

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T = 100
K = 20
sigma = 1
alpha = 0
set.seed(123)
f = seq(0.1,0.4,length=K) # Note 1/f should be the distance between peaks
beta = gamma = rep(0,K)
# Pick one frequency and see if the model can find it
choose = 4
beta[choose] = 2
gamma[choose] = 2
X = outer(2 * pi * 1:T, f, '*') # This creates a clever matrix of 2 * pi * t * f_k for every combination of t and f_k
mu = alpha + cos(X) %*% beta + sin(X) %*% gamma
y = rnorm(T, mu, sigma)
plot(1:T, y, type='l')
lines(1:T, mu, col='red')

# Look at the acf/pacf
acf(y)
pacf(y)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (t in 1:T) {
    y[t] ~ dnorm(mu[t], tau)
    mu[t] <- alpha + beta * cos( 2 * pi * t * f_k ) + gamma * sin( 2 * pi * t * f_k )
  }

  P = ( pow(beta, 2) + pow(gamma, 2) ) / 2

  # Priors
  alpha ~ dnorm(0.0,0.01)
  beta ~ dnorm(0.0,0.01)
  gamma ~ dnorm(0.0,0.01)
  tau <- 1/pow(sigma,2) # Turn precision into standard deviation
  sigma ~ dunif(0.0,100.0)
}
'

# Set up the data - run this repeatedly:
model_parameters =  c("P")
Power = rep(NA,K)

# A loop, but should be very fast
for (k in 1:K) {
  curr_model_data = list(y = y, T = T, f_k = f[k], pi = pi)

  model_run = jags(data = curr_model_data,
                   parameters.to.save = model_parameters,
                   model.file=textConnection(model_code),
                   n.chains=4, # Number of different starting positions
                   n.iter=1000, # Number of iterations
                   n.burnin=200, # Number of iterations to remove at start
                   n.thin=2) # Amount of thinning

  Power[k] = mean(model_run$BUGSoutput$sims.list$P)
}

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc

# Plot the posterior periodogram next to the time series
par(mfrow=c(2,1))
plot(1:T, y, type='l')
plot(f,Power,type='l')
abline(v=f[choose],col='red')
par(mfrow=c(1,1))

# Real example ------------------------------------------------------------

# Use the lynx data
library(rdatamarket)
lynx = as.ts(dmseries('http://data.is/Ky69xY'))
plot(lynx)

# Create some possible periodicities
periods = 5:40
K = length(periods)
f = 1/periods

# Run as before
for (k in 1:K) {
  curr_model_data = list(y = as.vector(lynx[,1]),
                         T = length(lynx),
                         f_k = f[k],
                         pi = pi)

  model_run = jags(data = curr_model_data,
                   parameters.to.save = model_parameters,
                   model.file=textConnection(model_code),
                   n.chains=4, # Number of different starting positions
                   n.iter=1000, # Number of iterations
                   n.burnin=200, # Number of iterations to remove at start
                   n.thin=2) # Amount of thinning

  Power[k] = mean(model_run$BUGSoutput$sims.list$P)
}

par(mfrow = c(2, 1))
plot(lynx)
plot(f, Power, type='l')
# Make this more useful by adding in a second axis showing periods
axis(side = 3, at = f, labels = periods)
par(mfrow=c(1, 1))
# Numbers seem to increase about every 10 years

# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks
# 1) Try experimenting with the simulated data to produce plots with different frequencies. Try versions where the true beta and gamma values have multiple different non-zero values. See if the periodogram picks them up
# 2) In the power plot above with the second axis it is sometimes helpful to have the periods at the bottom and the frequencies at the top (or not at all). Re-create the plot the other way round
# 3) (harder) In all of the above we have just stored the mean of the power for each fun. As we are fitting a Bayesian model we have the advantage that we have a posterior distribution of P for each frequency. See if you can find a way to plot it with uncertainty
