# Header ------------------------------------------------------------------

# P-spline binomial model in JAGS
# Andrew Parnell

# This file fits a spline binomial logistic regression model to data in JAGS, and produces predictions/forecasts

# Some boiler plate code to clear the workspace and load in required packages
rm(list=ls())
library(R2jags)
library(boot) # Logit functions
library(splines) # Useful for creating the B-spline basis functions
# Maths -------------------------------------------------------------------

# Notation:
# y(t): Response variable at time t, defined on continuous time. Here y is binary
# y: vector of all observations
# B: design matrix of spline basis functions
# beta; spline weights
# sigma_b: spline random walk parameter

# Likelihood:
# y ~ Binomial(1, inv.logit(B%*%beta))
# beta_j - beta_{j-1} ~ N(0, sigma_b)

# Priors
# sigma_b ~ cauchy(0, 10)

# Note: you could make this model a bit richer by instead having:
# y ~ binomial(1, p)
# logit(p) ~ N(B%*%beta, sigma)
# which would allow for over-dispersion (with parameter sigma)
# I've kept it as simple as possible here

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

# A function that uses the bs() function to generate the B-spline basis functions
# following Eilers and Marx 'Craft of smoothing' course. This bs_bbase() function
# is equivalent to the bbase() function available at http://statweb.lsu.edu/faculty/marx/

bs_bbase = function(x, xl = min(x), xr = max(x), nseg = 10, deg = 3){
  # Compute the length of the partitions
  dx = (xr - xl) / nseg
  # Create equally spaced knots
  knots = seq(xl - deg * dx, xr + deg * dx, by = dx)
  # Use bs() function to generate the B-spline basis
  get_bs_matrix = matrix(bs(x, knots = knots, degree = deg, Boundary.knots = c(knots[1], knots[length(knots)])), nrow = length(x))
  # Remove columns that contain zero only
  bs_matrix = get_bs_matrix[,-c(1:deg, ncol(get_bs_matrix):(ncol(get_bs_matrix) - deg))]
  
  return(bs_matrix)
}

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
set.seed(100)
T = 200 # Number of observations
x = sort(runif(T, 0, 10)) # Create some covariate values
# B = bbase(x)
B = bs_bbase(x, nseg = 30)
sigma_b = 1 # Parameters as above
beta = cumsum(c(0, rnorm(ncol(B)-1, 0, sigma_b)))
y = rbinom(T, 1, inv.logit(B%*%beta))
plot(x,y)
lines(x, inv.logit(B%*%beta), col = 'red') # True line

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (t in 1:T) {
    y[t] ~ dbinom(ilogit(inprod(B[t,], beta)), 1)
  }

  # RW prior on beta
  beta[1] ~ dnorm(0, 10^-2)
  for (i in 2:N_knots) {
    beta[i] ~ dnorm(beta[i-1], sigma_b^-2)
  }

  # Priors on beta values
  sigma_b ~ dt(0, 10^-2, 1)T(0,)
}
'

# Set up the data
model_data = list(T = T, y = y, B = B, N_knots = ncol(B))

# Choose the parameters to watch
model_parameters =  c("beta", "sigma_b")

# Run the model - can be slow
model_run = jags(data = model_data,
                   parameters.to.save = model_parameters,
                   model.file=textConnection(model_code))

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
print(model_run)

# Get the posterior betas and 50% CI
beta_post = model_run$BUGSoutput$sims.list$beta
beta_quantile = apply(beta_post, 2, quantile, prob = c(0.25, 0.5, 0.75))

# Plot the output with uncertainty bands
plot(x,y)
lines(x, inv.logit(B%*%beta), col = 'red') # True line
lines(x, inv.logit(B%*%beta_quantile[2,]), col = 'blue') # Predicted line
lines(x, inv.logit(B%*%beta_quantile[1,]), col = 'blue', lty = 2) # Predicted low
lines(x, inv.logit(B%*%beta_quantile[3,]), col = 'blue', lty = 2) # Predicted high
legend('topleft', c('True line',
                    'Posterior lines (with 50% CI)',
                    'Data'),
       lty = c(1, 1, -1),
       pch = c(-1, -1, 1),
       col = c('red', 'blue', 'black'))

# Create some new predictions on a grid of new values
# Needs to be in the same range as the previous values (if not you need to go back to the creation of B above)
x_new = seq(min(x), max(x), length = 1000)
# B_new = bbase(x_new, xl = min(x), xr = max(x))
B_new = bs_bbase(x_new, xl = min(x), xr = max(x), nseg = 30)
plot(x,y)
lines(x_new, inv.logit(B_new%*%beta_quantile[2,]), col = 'blue') # Beautifully smooth
lines(x_new, inv.logit(B_new%*%beta_quantile[1,]), col = 'blue', lty = 2) # Predicted low
lines(x_new, inv.logit(B_new%*%beta_quantile[3,]), col = 'blue', lty = 2) # Predicted high
legend('right', c('Posterior line on new grid (with 50% CI)',
                    'Data'),
       lty = c(1, -1),
       pch = c(-1, 1),
       col = c('blue', 'black'))

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory


# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks


