# Header ------------------------------------------------------------------

# JAGS for linear regression with missing explanatory variables and/or response


# Fitting a linear regression model in JAGS where some of the x and/or y values might be missing
# Andrew Parnell

# Some boiler plate code to clear the workspace, set the working directory, and load in required packages
rm(list = ls())
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation
# y_i = response variable, i = 1, ..., N
# X_i = vector of p explanatory variables
# alpha = intercept
# beta = vector of p regression coefficients
# sigma = residual standard deviation
# Some of the X and y values are NA - missing

# Likelihood
# y_i ~ N(alpha + x_i beta, sigma^2)

# Priors
# alpha ~ N(0, 10^2)
# beta_j ~ N(0, 10^2), j = 1, ..., p
# sigma ~ t+(0, 10^2)

# Simulate data -----------------------------------------------------------

set.seed(123) # Set the seed
N = 200 # Number of obs
p = 3 # Number of explanatory variables
N_miss_x = 100 # Number of missing x values
N_miss_y = 30 # Number of missing y values
X = X_obs = matrix(rnorm(N*p), ncol = p, nrow = N)
alpha = 3
beta = c(-1, 2, 1)
sigma = 0.5
y = y_obs = X%*%beta + rnorm(N, 0, sigma)

# Remove the missing values for a random subset
which_miss_x = sample(1:(N*p), size = N_miss_x)
which_miss_y = sample(1:(N), size = N_miss_y)
X_obs[which_miss_x] = NA
y_obs[which_miss_y] = NA

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dnorm(fits[i], sigma^-2)
    fits[i] = alpha + inprod(X[i,], beta)
  }
  # Priors for the missing x values
  for(k in 1:N_miss_x) {
    X[miss_row[k], miss_col[k]] ~ dnorm(0, 1) # Know that X is standard normal
  }

  # Priors
  alpha ~ dnorm(0, 100^-2)
  for (j in 1:p) {
    beta[j] ~ dnorm(0, 100^-2)
  }
  sigma ~ dt(0, 10^-1, 1)T(0, )
}
'

# Simulated results -------------------------------------------------------

# Run this model

# Trick is to get the miss_row and miss_col vectors set up right
which_miss = which(is.na(X_obs), arr.ind = TRUE)

# Set up the data
model_data = list(N = N,
                  y = y_obs[,1], # Simulated data created a matrix, but JAGS wants a vector
                  X = X_obs,
                  p = ncol(X_obs),
                  N_miss_x = nrow(which_miss),
                  miss_row = which_miss[,1],
                  miss_col = which_miss[,2])

# Choose the parameters to watch
model_parameters =  c("alpha", "beta", "sigma", "fits", "X")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code))

# See if it converged
plot(model_run) # Seemed to estimate the parameters well

# Now plot the predictions and see if it worked
fits = model_run$BUGSoutput$mean$fits

plot(y, fits)
abline(a = 0, b = 1, col = 'red')
# Looks ok

# What about the missing y values
plot(y[which_miss_y], fits[which_miss_y])
abline(a = 0, b = 1, col = 'red') # not so good but ok

# And the missing X values
X_fit = model_run$BUGSoutput$mean$X
plot(as.vector(X[which_miss_x]), as.vector(X_fit[which_miss_x]))
abline(a = 0, b = 1, col = 'red') # Does ok

