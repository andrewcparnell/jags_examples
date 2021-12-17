# Header ------------------------------------------------------------------

# Change point modelling in JAGS - a change point variance model
# Andrew Parnell

# This file implements a simple change point model on the variance in JAGS. This basic version has constant periods of variance with discontinuous jumps.

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls())
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y(t) = response variable observed at times t. In these files time can be discrete or continuous
# t_k = time of variance change point k - these are the key parameters to be estimated, k = 1, .., K - K is the number of change points
# mu = mean term
# gamma_k = variance of the data in each of the k periods
# sigma_sq_t = variance of each time point

# Likelihood:
# Top level likelihood is always:
# y(t) ~ normal(mu, sigma_sq_t)

# Then with one change point in the variance
# sigma_sq_t[t] = gamma[1] if t < t_1,
# or sigma_sq_t[t] = gamma[2] if t>=t_1

# For a model with 2 change points
# sigma_sq_t[t] = gamma[1] if t < t_1
# sigma_sq_t[t] = gamma[2] if t_1 <= t < t_2
# sigma_sq_t[t] = gamma[3] if t > t_2

# And so on if you want more change points

# To achieve this kind of model in jags we use the step function which works via:
# step(x) = 1 if x>0 or 0 otherwise.
# We can use it to pick out which side of the change point(s) we're on

# Priors
# mu ~ normal(0, 100)
# t_1 ~ uniform(t_min, t_max) # Restrict the change point to the range of the data
# gamma[k] ~ dunif(0, 100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model

# DCPR-1 model
T <- 100
mu <- 5
gamma = c(2, 10) # Variances before and after the change point
t_1 = 0.5 # Change point time value
set.seed(123)
t <- sort(runif(T))
sigma_sq <- rep(NA, T)
sigma_sq[t < t_1] <- gamma[1]
sigma_sq[t >= t_1] <- gamma[2]
y <- rnorm(T, mu, sqrt(sigma_sq))
plot(t, y)
abline(v = t_1, col = "blue") # Should see a big jump in variance at t_1

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data

# Code for one change point variance model
model_code_1CP <- "
model
{
  # Likelihood
  for(i in 1:T) {
    y[i] ~ dnorm(mu, tau[i])
    tau[i] <- 1 / sigma_sq[i]
    sigma_sq[i] <- gamma[J[i]]
    # This is the clever bit - only pick out the right change point when above t_1
    J[i] <- 1 + step(t[i] - t_1)
  }

  # Priors
  mu ~ dnorm(0, 100^-2)
  gamma[1] ~ dunif(0, 100)
  gamma[2] ~ dunif(0, 100)
  t_1 ~ dunif(t_min, t_max)
}
"

# Data for one change point model
jags_data <- list(t = t, y = y, T = T, t_min = min(t), t_max = max(t))

# Choose the parameters to watch
model_parameters <- c("t_1", "mu", "gamma")

# Run the model
model_run_1CP <- jags(
  data = jags_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code_1CP),
  n.chains = 4,
  n.iter = 1000,
  n.burnin = 200,
  n.thin = 2
)

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
print(model_run_1CP)
plot(model_run_1CP)

# Plot the data again with the estimated change point
t_1_mean <- model_run_1CP$BUGSoutput$sims.list$t_1
with(jags_data, plot(t, y))
abline(v = mean(t_1_mean), col = "red")

# Plot the parameter values with their true values (only possible for simulated data)
post <- model_run_1CP$BUGSoutput$sims.list
hist(post$gamma[,1]) # Estimated variance before the change point
abline(v = gamma[1], col = 'red')

hist(post$gamma[,2]) # Estimated variance after the change point
abline(v = gamma[2], col = 'red')

hist(post$t_1) # Estimated change point value
abline(v = t_1, col = 'red')
