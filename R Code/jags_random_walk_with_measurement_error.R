# Header ------------------------------------------------------------------

# Random walk models with measurement error
# Andrew Parnell

# In this code we fit some random walk type models to data

# Some boiler plate code to clear the workspace and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y(t) = response variable at time t, t = 1,...,T
# mu(t) = latest mean parameter (smooth)
# delta = drift parameter (optional)
# sigma = residual standard deviation
# sigma_delta = random walk standard deviation

# Likelihood:
# Order 1: y(t) ~ N(mu(t),sigma^2)
# mu(t) - mu(t-1) ~ N(delta, sigma_delta^2)
# Prior:
# sigma ~ unif(0,100) - vague
# sigma_delta ~ unif(0, 10) # Can play with this
# delta ~ dnorm(0, 100) - vague

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
set.seed(123)
T <- 100
sigma_delta <- 10
delta <- 1
sigma <- 20
t <- 1:T
mu <- cumsum(rnorm(T, delta, sigma_delta))
y <- rnorm(T, mu, sigma)

plot(t, y)
lines(t, mu)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
# Note: running the differencing offline here as part of the data step
model_code <- "
model
{
  # Likelihood
  for (t in 1:N_T) {
    y[t] ~ dnorm(mu[t], sigma^-2)
  }
  mu[1] ~ dnorm(0, 100^-2)
  for (t in 2:N_T) {
    mu[t] ~ dnorm(mu[t-1] + delta, sigma_delta^-2)
  }

  # Priors
  delta ~ dnorm(0, 100^-2)
  sigma ~ dunif(0, 100)
  sigma_delta ~ dunif(0, 100)
}
"

# Set up the data
model_data <- list(y = y, N_T = T)

# Choose the parameters to watch
model_parameters <- c("mu", "delta", "sigma")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
) # Amount of thinning

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
plot(model_run)

# Show the output
plot(t, y)
lines(t, mu)
lines(t, model_run$BUGSoutput$mean$mu, col = 'red')

