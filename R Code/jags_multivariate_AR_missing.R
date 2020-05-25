# Header ------------------------------------------------------------------

# Multivariate Autoregressive models, commonly known as Vector AR (VAR) models
# This one allows for missing values in different parts of the response
# Andrew Parnell

# The VAR model is a multivariate extension of the standard AR(p) model. In this code I just fit the VAR(1) model but it is easily extended to VAR(p)

# Some boiler plate code to clear the workspace and load in required packages
rm(list=ls())
library(R2jags)
library(MASS) # Used to generate MVN samples

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation
# y_t = multivariate response variable at time t, t=1,...,T. Each y_t is a vetor of dimension k
# A = k-vector of constants
# Phi = k by k matrix of AR coefficients - these are our most important parameters
# e_t = k-vector of residuals
# sigma_k^2 = residual standard deviation for dimension k
# Note some of the k dimensions at each time t

# Likelihood
# y_t = A + Phi * y_{t-1} + e_t with e_t ~ MVN(0, sigma_k^2 I)
# or
# y_kt ~ N(A + Phi * y_{t-1}, sigma^2 I)

# Prior
# A[k] ~ normal(0, 100)
# Phi[j,k] ~ normal(0, 100)
# sigma^2 ~ Inverse Gamma(1, 1)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T = 100
K = 3
set.seed(123)
sigma = runif(K, 0.5, 1.5)
#Phi = matrix(c(0.6, 0.2, 0.2, 0.8), 2, 2)
Phi = runif(K, 0.1, 0.9) * diag(K)
A = matrix(1:K, nrow = K, ncol = 1)
y = matrix(NA, T, K)
y[1,] = solve(diag(K) - Phi) %*% A # Long term average of process
for(t in 2:T) {
  y[t,] = mvrnorm(1, A + Phi %*% y[t-1,], sigma^2 * diag(K))
}

# Now add in missingness
miss_prob = 0.2
which_miss = matrix(rbinom(T*K, size = 1, prob = miss_prob),
                    ncol = K, nrow = T)
for(i in 1:K) y[which_miss[,i] == 1,i] = NA

# Plot the output
par(mfrow = c(K, 1))
for(i in 1:K) plot(1:T, y[,i], type = 'l')
par(mfrow = c(1, 1))

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for(k in 1:K) {
    y[1, k] ~ dnorm(mu[1, k], sigma[k]^-2)
    mu[1,k] ~ dnorm(0, 100^-2)
    for (t in 2:T) {
      y[t, k] ~ dnorm(mu[t, k], sigma[k]^-2)
      mu[t,k] <- A[k] + Phi[k,] %*% y[t-1,]
    }
  }

  # Priors
  for(k in 1:K) {
    A[k] ~ dnorm(0, 100^-2)
    sigma[k] ~ dunif(0, 100)
    Phi[k,k] ~ dnorm(0, 1^-2)
    for(j in (k+1):K) {
      Phi[k,j] ~ dnorm(0, 1^-2)
      Phi[j,k] ~ dnorm(0, 1^-2)
    }
  }
}
'

# Set up the data
model_data = list(T = T, K = K, y = y)

# Choose the parameters to watch
model_parameters =  c("A", "Phi", "sigma", "mu")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code),
                 n.iter = 10000,
                 n.burnin = 2000,
                 n.thin = 8)

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
print(model_run) # Results look pretty good
plot(model_run)
stop()

# Have a look to see if it predicted the missing values well
mu_mean = model_run$BUGSoutput$mean$mu
# Plot the output
par(mfrow = c(K, 1))
for(i in 1:K) {
  plot(1:T, y[,i], type = 'l')
  lines(1:T, mu_mean[,i], col = 'blue') # Looks like it's predicted at a lag but ok
}
par(mfrow = c(1, 1))

# Real example ------------------------------------------------------------

# Can we fit a vector AR model to both the sea level and global temperature
# series?
hadcrut = read.csv('https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/hadcrut.csv')
sea_level = read.csv('https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/church_and_white_global_tide_gauge.csv')
head(hadcrut)
head(sea_level)

# Correct the sea level ages
sea_level$Year2 = sea_level$year_AD-0.5

# Merge them together
bivariate_data = merge(hadcrut, sea_level, by.x='Year', by.y='Year2')

# Plot the two of them together
par(mfrow=c(2,1))
with(bivariate_data, plot(Year, Anomaly, type='l'))
with(bivariate_data, plot(Year, sea_level_m, type='l'))
par(mfrow=c(1,1))

# Perhaps run on differences
par(mfrow=c(3,1))
with(bivariate_data, plot(Year[-1], diff(Anomaly), type='l'))
with(bivariate_data, plot(Year[-1], diff(sea_level_m), type='l'))
with(bivariate_data, plot(diff(Anomaly), diff(sea_level_m)))
par(mfrow=c(1,1))

# Create the data
real_data = with(bivariate_data,
                 list(T = nrow(bivariate_data)-1,
                      y = apply(bivariate_data[,c('Anomaly', 'sea_level_m')],2,'diff'),
                      K = 2))

# Run the model
real_data_run = jags(data = real_data,
                     parameters.to.save = model_parameters,
                     model.file=textConnection(model_code),
                     n.iter=10000,
                     n.burnin=2000,
                     n.thin=8)

# Plot output
print(real_data_run)
plot(real_data_run)

# Let's create some joint predictions off into the future
n_forecast = 10

real_data_future = with(bivariate_data,
                 list(T = nrow(bivariate_data) + n_forecast - 1,
                      y = rbind(as.matrix(apply(bivariate_data[,c('Anomaly', 'sea_level_m')],2,'diff')), matrix(NA, ncol=2, nrow=n_forecast)),
                      K = 2))

# Choose the parameters to watch
model_parameters =  c("y")

real_data_run_future = jags(data = real_data_future,
                            parameters.to.save = model_parameters,
                            model.file=textConnection(model_code),
                            n.iter=10000,
                            n.burnin=2000,
                            n.thin=8)

plot(real_data_run_future)

y_future_pred = real_data_run_future$BUGSoutput$sims.list$y
y_future_med = apply(y_future_pred,c(2,3),'median')
year_all = c(bivariate_data$Year[-1],2010:(2010+n_forecast))

# Create plots
par(mfrow=c(2,1))
plot(year_all[-1]-1, bivariate_data$Anomaly[1]+cumsum(y_future_med[,1]), col='red', type='l')
with(bivariate_data, lines(Year, Anomaly))
plot(year_all[-1]-1, bivariate_data$sea_level_m[1]+cumsum(y_future_med[,2]), col='red', type='l')
with(bivariate_data, lines(Year, sea_level_m))
par(mfrow=c(1,1))


