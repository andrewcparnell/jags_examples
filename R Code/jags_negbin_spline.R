# Header ------------------------------------------------------------------

# P-spline model for over-dispersed count data in JAGS
# Andrew Parnell

# This file fits a negative binomial spline regression model to data in JAGS, and produces predictions/forecasts
# Some of the negbin stuff taken from here: https://georgederpa.github.io/teaching/countModels.html

# Some boiler plate code to clear the workspace and load in required packages
rm(list=ls())
library(R2jags)
library(MASS) # Useful for rnegbin function

# Maths -------------------------------------------------------------------

# Notation:
# y(t): Response variable at time t, defined on continuous time and is a count
# y: vector of all observations
# B: design matrix of spline basis functions
# beta; spline weights
# theta; Overdispersion (OD) parameter of the NegBin
# sigma_b: spline random walk parameter

# Likelihood:
# y ~ Negbin(B%*%beta, theta) # where the first argument is the mean, and the second is the OD
# beta_j - beta_{j-1} ~ N(0, sigma_b^2)

# Priors
# theta_inv ~ half-cauchy(0, 10)
# sigma_b ~ half-cauchy(0, 10)

# Useful function ---------------------------------------------------------

# These functions create the B-spline basis functions
# They are taken from the Eilers and Marx 'Craft of smoothing' course
# http://statweb.lsu.edu/faculty/marx/
tpower = function(x, t, p)
  # Truncated p-th power function
  return((x - t) ^ p * (x > t))
bbase = function(x, xl = min(x), xr = max(x), nseg = 30, deg = 3){
  # Construct B-spline basis
  dx = (xr - xl) / nseg
  knots = seq(xl - deg * dx, xr + deg * dx, by = dx)
  P = outer(x, knots, tpower, deg)
  n = dim(P)[2]
  D = diff(diag(n), diff = deg + 1) / (gamma(deg + 1) * dx ^ deg)
  B = (-1) ^ (deg + 1) * P %*% t(D)
  return(B)
}


# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
set.seed(123)
N = 100 # Number of observations
x = sort(runif(N, 0, 10)) # Create some covariate values
B = bbase(x)
sigma_b = 0.5 # Parameters as above
theta = 10
beta = cumsum(c(1, rnorm(ncol(B)-1, 0, sigma_b)))
y = rnegbin(N, mu = exp(B%*%beta), theta = theta)
plot(x,y)
lines(x, exp(B%*%beta), col = 'red') # True line

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dnegbin(p[i], theta)
    p[i] <- theta/(theta+lambda[i])
    log(lambda[i]) <- mu[i]
    mu[i] <- inprod(B[i,],beta)
  }

  # RW prior on beta
  beta[1] ~ dnorm(0, 10^-2)
  for (i in 2:N_knots) {
    beta[i] ~ dnorm(beta[i-1], sigma_b^-2)
  }

  # Priors on beta values
  theta = 1/theta_inv
  theta_inv ~ dt(0, 10^-2, 1)T(0,)
  sigma_b ~ dt(0, 10^-2, 1)T(0,)

  # Create predictions and prediction intervals as required
  for(i in 1:N_pred) {
    y_pred[i] ~ dnegbin(p_pred[i], theta)
    p_pred[i] <- theta/(theta+lambda_pred[i])
    log(lambda_pred[i]) <- mu_pred[i]
    mu_pred[i] <- inprod(B_pred[i,],beta)
  }

}
'

# Suppose we want some predictions on a regular grid
N_pred = 200
x_pred = seq(min(x), max(x), length = N_pred)
B_pred = bbase(x_pred, xl = min(x), xr = max(x))

# Set up the data
model_data = list(N = N, y = y, B = B, N_knots = ncol(B), B_pred = B_pred, N_pred = N_pred)

# Choose the parameters to watch
model_parameters =  c("beta", "theta", "sigma_b", "y_pred")

# Run the model - can be slow
model_run = jags(data = model_data,
                   parameters.to.save = model_parameters,
                   model.file=textConnection(model_code))

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
plot(model_run)

# Get the posterior betas and 50% CI
y_post = model_run$BUGSoutput$sims.list$y_pred
y_quantile = apply(y_post, 2, quantile, prob = c(0.25, 0.5, 0.75))

# Plot the output with uncertainty bands
plot(x,y)
lines(x_pred, exp(B_pred%*%beta), col = 'red') # True line
lines(x_pred, y_quantile[2,], col = 'blue') # Predicted line
lines(x_pred, y_quantile[1,], col = 'blue', lty = 2) # Predicted low
lines(x_pred, y_quantile[3,], col = 'blue', lty = 2) # Predicted high
legend('topleft', c('True line',
                    'Posterior lines (with 50% PI)',
                    'Data'),
       lty = c(1, 1, -1),
       pch = c(-1, -1, 1),
       col = c('red', 'blue', 'black'))


# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory

# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks


