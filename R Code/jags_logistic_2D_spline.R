# Header ------------------------------------------------------------------

# A 2D logistic regression P-spline model in JAGS
# Possibly suitable for spatial data analysis
# Andrew Parnell

# This file fits a 2D spline logistic regression model to data in JAGS, and produces predictions/forecasts

# Some boiler plate code to clear the workspace and load in required packages
rm(list=ls())
library(R2jags)
library(MASS) # Useful for mvrnorm function
library(akima) # Useful for 2D interpolation
library(boot) # Logit functions

# Maths -------------------------------------------------------------------

# Notation:
# y(x1, x2): Response variable at covariate values x1 and x2, defined on continuous space. Y is assumed binary
# y: vector of all observations
# B: design matrix of spline basis functions - needs to be a tensor product over 2 dimensions
# beta; spline weights - again needs to be a vector
# sigma: residual standard deviation parameter (sometimes known in the GP world as the nugget)
# sigma_b: spline 2D random walk standard deviation parameter

# Likelihood:
# y ~ Bernoulli(inv.logit(B%*%beta))
# beta_j ~ N(0, sigma_b^2)

# Priors
# sigma_b ~ cauchy(0, 10)

# Useful function ---------------------------------------------------------

# These functions create the B-spline basis functions
# They are taken from the Eilers and Marx 'Craft of smoothing' course
# http://statweb.lsu.edu/faculty/marx/
tpower = function(x, t, p)
  # Truncated p-th power function
  return((x - t) ^ p * (x > t))
bbase = function(x, xl = min(x), xr = max(x), nseg = 10, deg = 3){
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
N = 200 # Number of observations
x1 = runif(N, 0, 10) # Create some 2D covariate values
x2 = runif(N, 0, 10) # Create some 2D covariate values

# Create a vector of all the x values
x = cbind(x1, x2)

# Create two individual basis function matrices
B1 = bbase(x[,1], xl = 0, xr = 10) # Put ranges on these with xl and xr for later ease of interpolation
B2 = bbase(x[,2], xl = 0, xr = 10)

# Create the matrix which is now going to be each column of B1 multiplied by each column of B2
# There's perhaps a more elegant way of doing this
B = matrix(NA, ncol = ncol(B1)*ncol(B2), nrow = N)
count = 1
for(i in 1:ncol(B1)) {
  for(j in 1:ncol(B2)) {
    B[,count] = B1[,i] * B2[,j]
    count = count + 1
  }
}
# If required, plot some of the basis functions
# for(i in 1:ncol(B)) {
#   plot(x[,1], x[,2], cex = B[,i]/max(B[,i]), pch = 19)
#   Sys.sleep(0.1)
# }
sigma_b = 3 # Parameters as above
beta = rnorm(ncol(B), 0, sigma_b)
mu = B%*%beta
y = rbinom(N, size = 1, prob = inv.logit(mu))

# Plot the underlying mean surface
mu_interp = interp(x1, x2, mu)
with(mu_interp, contour(x, y, z, col = 'red'))

# Add in the data points too if reuquired
points(x[,1], x[,2], col = as.factor(y), pch = 19)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dbern(mu[i])
    logit(mu[i]) = inprod(B[i,], beta)
  }

  # Prior on beta
  for (i in 1:N_knots) {
    beta[i] ~ dnorm(0, sigma_b^-2)
  }

  # Priors on beta values
  sigma_b ~ dt(0, 10^-2, 1)T(0,)
}
'

# Set up the data
model_data = list(N = N, y = y, B = B, N_knots = ncol(B))

# Choose the parameters to watch
model_parameters =  c("beta", "sigma_b")

# Run the model - can be slow
model_run = jags(data = model_data,
                   parameters.to.save = model_parameters,
                   model.file=textConnection(model_code))
stop()

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
print(model_run)
plot(model_run)

# Get the posterior betas and 50% CI
beta_post = model_run$BUGSoutput$sims.list$beta
beta_median = apply(beta_post, 2, median)

# Plot the output with uncertainty bands

# First plot the data
plot(x[,1], x[,2], col = as.factor(y), pch = 19)

# Now plot the true mean surface
with(mu_interp, contour(x, y, z, col = 'red', add = TRUE))

# Now plot the estimated median surface
mu_pred = B%*%beta_median
mu_pred_interp = interp(x1, x2, mu_pred)
with(mu_pred_interp, contour(x, y, z, col = 'blue', add = TRUE)) # Red and blue contour lines should look similar

# Finally create some new predictions on a grid of new values
# Needs to be in the same range as the previous values (if not you need to go back to the creation of B above)
n_grid = 50
x1_new = seq(0, 10, length = n_grid)
x2_new = seq(0, 10, length = n_grid)
x_new = expand.grid(x1_new, x2_new)

# Create new B matrix
B1_new = bbase(x_new[,1], xl = 0, xr = 10) # Put ranges on these with xl and xr for later ease of interpolation
B2_new = bbase(x_new[,2], xl = 0, xr = 10)

# Create the matrix which is now going to be each column of B1 multiplied by each column of B2
# There's perhaps a more elegant way of doing this
B_new = matrix(NA, ncol = ncol(B1_new)*ncol(B2_new), nrow = n_grid^2)
count = 1
for(i in 1:ncol(B1_new)) {
  for(j in 1:ncol(B2_new)) {
    B_new[,count] = B1_new[,i] * B2_new[,j]
    count = count + 1
  }
}

# Plot the new interpolated predictions on top
mu_interp_pred = B_new%*%beta_median
mu_pred_interp_2 = interp(x_new[,1], x_new[,2], mu_interp_pred)
with(mu_pred_interp_2, contour(x, y, z, col = 'green', add = TRUE)) # Red and blue contour lines should look similar

# Or just plot them separately
with(mu_pred_interp_2, image(x, y, z)) # Red and blue contour lines should look similar
points(x[,1], x[,2], col = as.factor(y), pch = 19) # Looks about right


# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory


# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks


