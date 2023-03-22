# Header ------------------------------------------------------------------

# P-spline model in JAGS with robust specification of the roughness of the
# penalty. This version includes a second spline on the variance to model
# heteroskedasticity

# Mateus Maia & Andrew Parnell

# This file fits a spline regression model to data in JAGS, adding a roughness
# parameter into the precision of the splines to yielding a more robust
# hyperparameter specification

# Some boiler plate code to clear the workspace and load in required packages
rm(list = ls())
library(R2jags)
library(MASS) # Useful for mvrnorm function
library(splines) # Useful for creating the B-spline basis functions
library(boot) # For real example

# Maths -------------------------------------------------------------------

# Notation:
# y: vector of all observations
# B: design matrix of spline basis functions
# beta; spline weights
# tau; residual precision
# tau_beta; spline random walk precision parameter
# delta; the roughness parameter for tau_b prior

# Likelihood
# y ~ N(B%*%beta, tau_s^-1)
# beta[j] ~ N (beta[j-1],tau_b^-1)
# log(tau_s) ~ N(B %*% beta_tau, tau)
# beta_tau[j] ~ N(beta[j-1],tau_bs^-1)

# Priors
# tau ~ gamma(a_tau, d_tau)
# tau_b ~ gamma(0.5*nu1, 0.5*delta1*nu1)
# tau_bs ~ gamma(0.5*nu2, 0.5*delta2*nu2)
# delta[k] ~ gamma(a_delta,d_delta)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
set.seed(42)
N <- 200 # Number of observations
x <- sort(runif(N, 0, 10)) # Create some covariate values
nIknots_sim <- 100
knots <- quantile(x, seq(0, 1, length.out = nIknots_sim + 2))[-c(1, nIknots_sim + 2)] # Setting the same number of observations by node
B <- splines::ns(x = x, knots = knots, intercept = FALSE)
tau_b <- 1 # Parameters as above
tau <- 10
beta <- cumsum(c(1, rnorm(ncol(B) - 1, 0, sqrt(tau_b^-1))))
y <- rnorm(N, mean = B %*% beta, sd = sqrt(tau^-1))

# Setting the nIknots and the basis that will be used
nIknots <- 100
knots <- quantile(x, seq(0, 1, length.out = nIknots + 2))[-c(1, nIknots + 2)] # Setting the same number of observations by node
B_train <- splines::ns(x = x, knots = knots, intercept = FALSE)
plot(x, y)


# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model {
  # Likelihood
  for (t in 1:N) {
    y[t] ~ dnorm(inprod(B[t,], beta), exp(log_tau_s[t]))
    y_pred[t] ~ dnorm(inprod(B[t,], beta), exp(log_tau_s[t]))
    log_tau_s[t] ~ dnorm(inprod(B[t,], beta_s), tau)
  }

  # RW prior on beta
  beta[1] ~ dnorm(0, tau_b_0)
  for (i in 2:N_knots) {
    beta[i] ~ dnorm(beta[i-1], tau_b)
  }

  # Prior on beta_s
  beta_s[1] ~ dnorm(0, tau_b_0)
  for (i in 2:N_knots) {
    beta_s[i] ~ dnorm(beta_s[i-1], tau_bs)
  }

  # Priors on beta values
  tau ~ dgamma(a_tau, d_tau)
  tau_b ~ dgamma(0.5 * nu, 0.5 * delta1 * nu)
  tau_bs ~ dgamma(0.5 * nu, 0.5 * delta2 * nu)
  delta1 ~ dgamma(a_delta, d_delta)
  delta2 ~ dgamma(a_delta, d_delta)
}
"

# Set up the data
model_data <- list(
  N = nrow(B_train),
  y = y,
  B = B_train,
  N_knots = ncol(B_train),
  a_tau = 1,
  d_tau = 1,
  a_delta = 0.0001, # Default values used in Jullion, A. and Lambert, P., 2007.
  d_delta = 0.0001, # Default values used in Jullion, A. and Lambert, P., 2007.
  tau_b_0 = 0.1,
  nu = 2
) # Default values used in Jullion, A. and Lambert, P., 2007.

# Choose the parameters to watch
model_parameters <- c("beta", "beta_s", "tau", "tau_b", "tau_bs", "delta1",
                      "delta2")

# Run the model - can be slow
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code)
)
plot(model_run)

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
# print(model_run)

# Get the posterior betas and 50% CI
beta_post <- model_run$BUGSoutput$sims.list$beta
beta_quantile <- apply(beta_post, 2, quantile, prob = c(0.25, 0.5, 0.75))

# New prediction
# Plot the output with uncertainty bands
plot(x, y)
lines(x, B_train %*% beta, col = "red") # True line
lines(x, B_train %*% beta_quantile[2, ], col = "blue") # Predicted line
lines(x, B_train %*% beta_quantile[1, ], col = "blue", lty = 2) # Predicted low
lines(x, B %*% beta_quantile[3, ], col = "blue", lty = 2) # Predicted high
legend("topleft", c(
  "True line",
  "Posterior lines (with 50% CI)",
  "Data"
),
lty = c(1, 1, -1),
pch = c(-1, -1, 1),
col = c("red", "blue", "black")
)

# Create some new predictions on a grid of new values
# Needs to be in the same range as the previous values (if not you need to go back to the creation of B above)
x_new <- seq(min(x), max(x), length = 1000)
knots <- quantile(x, seq(0, 1, length.out = nIknots + 2))[-c(1, nIknots + 2)]
B_new <- splines::ns(x = x_new, knots = knots, intercept = FALSE)
plot(x, y)
lines(x_new, B_new %*% beta_quantile[2, ], col = "blue") # Beautifully smooth
lines(x_new, B_new %*% beta_quantile[1, ], col = "blue", lty = 2) # Predicted low
lines(x_new, B_new %*% beta_quantile[3, ], col = "blue", lty = 2) # Predicted high
lines(x, B_train %*% beta, col = "red") # True line
legend("topright", c(
  "True line",
  "Posterior lines (with 50% CI)",
  "Data"
),
lty = c(1, 1, -1),
pch = c(-1, -1, 1),
col = c("red", "blue", "black")
)

# Real data example -------------------------------------------------------

# Use the motorbike data from boot
# Use times as the covariate and accel as the response
x <- motor$times
y <- motor$accel
plot(x, y)

# Set up the basis functions
nIknots <- 100
knots <- quantile(x, seq(0, 1, length.out = nIknots + 2))[-c(1, nIknots + 2)] # Setting the same number of observations by node
B_train <- splines::ns(x = x, knots = knots, intercept = FALSE)

# Set up the data
real_data <- list(
  N = nrow(B_train),
  y = y,
  B = B_train,
  N_knots = ncol(B_train),
  a_tau = 1,
  d_tau = 1,
  a_delta = 0.0001, # Default values used in Jullion, A. and Lambert, P., 2007.
  d_delta = 0.0001, # Default values used in Jullion, A. and Lambert, P., 2007.
  tau_b_0 = 0.1,
  nu = 2
) # Default values used in Jullion, A. and Lambert, P., 2007.

# Choose the parameters to watch
real_parameters <- c("beta", "beta_s", "tau", "tau_b", "tau_bs", "delta1", "delta2",
                     "y_pred")

# Run the model - can be slow
real_run <- jags(
  data = real_data,
  parameters.to.save = real_parameters,
  model.file = textConnection(model_code)
)
plot(real_run)

# Get the solutions
beta_post <- real_run$BUGSoutput$sims.list$beta
beta_quantile <- apply(beta_post, 2, quantile, prob = c(0.25, 0.5, 0.75))

# Plot answers on a grid
x_new <- seq(min(x), max(x), length = 1000)
knots <- quantile(x, seq(0, 1, length.out = nIknots + 2))[-c(1, nIknots + 2)]
B_new <- splines::ns(x = x_new, knots = knots, intercept = FALSE)
plot(x, y)
lines(x_new, B_new %*% beta_quantile[2, ], col = "blue") # Beautifully smooth
lines(x_new, B_new %*% beta_quantile[1, ], col = "blue", lty = 2) # Predicted low
lines(x_new, B_new %*% beta_quantile[3, ], col = "blue", lty = 2) # Predicted high
legend("topleft", c(
  "Posterior lines (with 50% CI)",
  "Data"
),
lty = c(1, -1),
pch = c(-1, 1),
col = c("blue", "black")
)

# Posterior predictive - check to see if the model is calibrated
y_post <- real_run$BUGSoutput$sims.list$y_pred
y_med <- apply(y_post, 2, 'quantile', 0.5)
y_low <- apply(y_post, 2, 'quantile', 0.25)
y_high <- apply(y_post, 2, 'quantile', 0.75)
plot(y, y_med, ylim = range(c(y_med, y_low, y_high)))
abline(a = 0, b = 1)
for (i in 1:length(y)) {
  lines(c(y[i], y[i]), c(y_low[i], y_high[i]))
}

