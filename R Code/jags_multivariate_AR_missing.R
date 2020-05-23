# Header ------------------------------------------------------------------

# Multivariate Autoregressive models, commonly known as Vector AR (VAR) models
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
# Sigma = k by k matrix of residual variance and co-variances

# Likelihood
# y_t = A + Phi * y_{t-1} + e_t with e_t ~ MVN(0, Sigma)
# or
# y_t ~ MVN(A + Phi * y_{t-1}, Sigma)

# Prior
# A[k] ~ normal(0, 100)
# Phi[j,k] ~ normal(0, 100)
# Sigma ~ Inverse Wishart(I, k+1)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T = 100
k = 2
Sigma = matrix(c(1, 0.2, 0.2, 1), 2, 2)
Phi = matrix(c(0.6, 0.2, 0.2, 0.8), 2, 2)
A = matrix(c(0, 2), 2, 1)
y = matrix(NA, T, k)
y[1,] = A
set.seed(123)
for(t in 2:T) {
  y[t,] = mvrnorm(1, A + Phi %*% y[t-1,], Sigma)
}

# Plot the output
par(mfrow = c(2, 1))
plot(1:T, y[,1], type = 'l')
plot(1:T, y[,2], type = 'l')
par(mfrow = c(1, 1))

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (t in 2:T) {
    y[t, ] ~ dmnorm(mu[t, ], Sigma.Inv)
    mu[t, 1:k] <- A + Phi %*% y[t-1,]
  }
  Sigma.Inv ~ dwish(I, k+1)
  Sigma <- inverse(Sigma.Inv)

  # Priors
  for(i in 1:k) {
    A[i] ~ dnorm(0, 0.01)
    Phi[i,i] ~ dunif(-1, 1)
    for(j in (i+1):k) {
      Phi[i,j] ~ dunif(-1,1)
      Phi[j,i] ~ dunif(-1,1)
    }
  }
}
'

# Set up the data
model_data = list(T = T, k = k, y = y, I = diag(k))

# Choose the parameters to watch
model_parameters =  c("A", "Phi", "Sigma")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code),
                 n.chains=4, # Number of different starting positions
                 n.iter=10000, # Number of iterations
                 n.burnin=2000, # Number of iterations to remove at start
                 n.thin=8) # Amount of thinning

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
print(model_run) # Results look pretty good

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
                      k = 2,
                      I = diag(2)))

# Run the model
real_data_run = jags(data = real_data,
                     parameters.to.save = model_parameters,
                     model.file=textConnection(model_code),
                     n.chains=4,
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
                      k = 2,
                      I = diag(2)))

# Choose the parameters to watch
model_parameters =  c("y")

real_data_run_future = jags(data = real_data_future,
                     parameters.to.save = model_parameters,
                     model.file=textConnection(model_code),
                     n.chains=4,
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


