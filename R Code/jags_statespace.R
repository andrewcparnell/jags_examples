# Header ------------------------------------------------------------------

# State space modelling in jags
# Andrew Parnell

# State space models are awesome! This code fits the most basic of state space models, a linear Gaussian version in discrete time.
# This model is often known as the Kalman filter
# Much richer, more interesting, and more useful versions are possible with a few simple tweaks
# There are lots of other names for this kind of model, including Hidden Markov Model (HMM), Dynamic Linear Models (DLMs) and Bayesian Networks (BN)


# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation
# y_t = response variable at time t, t=1,...,T
# x_t = latent time series of interest, defined at the same times as y
# alpha_y = intercept for state equation
# beta_y = slope for state equation
# sigma_y = residual standard deviation for state equation
# sigma_x = residual standard deviation for evolution equation

# Likelihood:
# y_t ~ N(alpha_y + beta_y * x_t, sigma_y) - this is the STATE equation
# x_t ~ N(x_{t-1}, sigma_x) - this is the EVOLUTION equation

# Note that this is not a well-defined model since you can swap the sign of x_t and of beta_y and still get exactly the same model
# In the below I'm going assume that alpha_y and beta_y are known and part of the data.

# Prior - these should be informative as this is not a well-defined model
# sigma_y ~ uniform(sig_y_lo, sig_y_hi)
# sigma_x ~ uniform(sig_x_lo, sig_x_hi)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T <- 100
sigma_x <- 1
sigma_y <- 1
alpha_y <- 3
beta_y <- 2
x <- y <- rep(NA, T)
x[1] <- 0
set.seed(123)
for (t in 2:T) x[t] <- rnorm(1, x[t - 1], sigma_x)
y <- rnorm(T, mean = alpha_y + beta_y * x, sigma_y)

plot(1:T, y)
lines(1:T, x, col = "red")
# Key task is to find x given y, and also to find values of alpha_y, beta_y, etc
# The former is called state estimation, the latter parameter estimation

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (t in 1:T) {
    y[t] ~ dnorm(alpha_y + beta_y * x[t], tau_y)
  }
  x[1] ~ dnorm(0, 0.01)
  for (t in 2:T) {
    x[t] ~ dnorm(x[t-1], tau_x)
  }

  # Priors
  tau_y <- 1/pow(sigma_y, 2)
  sigma_y ~ dunif(sig_y_lo, sig_y_hi)
  tau_x <- 1/pow(sigma_x, 2)
  sigma_x ~ dunif(sig_x_lo, sig_x_hi)
  }
"

# Set up the data - need the values for the hyper parameters her
model_data <- list(T = T, y = y, alpha_y = alpha_y, beta_y = beta_y, sig_y_lo = 0.0, sig_y_hi = 100, sig_x_lo = 0.0, sig_x_hi = 100)

# Choose the parameters to watch
model_parameters <- c("sigma_y", "sigma_x", "x")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4, # Number of different starting positions
  n.iter = 10000, # Number of iterations
  n.burnin = 2000, # Number of iterations to remove at start
  n.thin = 8
) # Amount of thinning

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
plot(model_run)
print(model_run)

# Look at the correlation between the parameters in the posterior
cor(model_run$BUGSoutput$sims.matrix[, 1:5]) # - still some very strong correlations

# Plot the latent x variables
x_mean <- apply(model_run$BUGSoutput$sims.list$x, 2, "mean")
par(mfrow = c(2, 1))
plot(1:T, y)
legend("topleft",
  legend = c("data", "truth", "estimated"),
  lty = c(-1, 1, 1),
  pch = c(1, -1, -1),
  col = c("black", "red", "blue"),
  horiz = TRUE
)
plot(1:T, x, col = "red", type = "l")
lines(1:T, x_mean, col = "blue")
par(mfrow = c(1, 1))

# Real example ------------------------------------------------------------

# Palaeoclimate reconstruction (fake data)
# Often in palaeoclimate reconstruction you have observed y (a proxy measurement)
# for a long period (e.g. 1000 years) and also observed x but only for a subset of that period
# e.g. the last 150 years. The task is to estimate x for the full period (state estimation),
# and to work out the relationship between the proxy and the climate (parameter estimation)

# In this example y is a standardised average tree ring width for 1000AD to 2015AD
# and x is the mean NH temperature for the subset 1880 to 2015
palaeo <- read.csv("https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/palaeo.csv")
par(mfrow = c(2, 1))
with(palaeo, plot(year, proxy))
with(palaeo, plot(year, temp)) # Only available for a subset
par(mfrow = c(1, 1))

# Estimate the parameters
pars <- lm(proxy ~ temp, data = palaeo)

# Set up the data - need the values for the hyper parameters her
real_data <- list(
  T = nrow(palaeo),
  y = palaeo$proxy,
  alpha_y = coef(pars)[1],
  beta_y = coef(pars)[2],
  sig_y_lo = 0.0,
  sig_y_hi = 100,
  sig_x_lo = 0.0,
  sig_x_hi = 100
)

# Run the model
real_model_run <- jags(
  data = real_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4, # Number of different starting positions
  n.iter = 10000, # Number of iterations
  n.burnin = 2000, # Number of iterations to remove at start
  n.thin = 8
) # Amount of thinning

# Look at the output
plot(real_model_run)

# Plot the latent x variables
x_mean <- apply(real_model_run$BUGSoutput$sims.list$x, 2, "mean")
with(palaeo, plot(year, temp)) # Only available for a subset
lines(palaeo$year, x_mean, col = "blue")
legend("topleft",
  legend = c("data", "estimated"),
  lty = c(-1, 1),
  pch = c(1, -1),
  col = c("black", "blue")
)

# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks
# 1) Try re-simulating the data iwth different values for sigma_x and sigma_y. Do certain combinations (e.g. large sigma_x, small sigma_y) cause problems for the convergence of the algorithm? What happens if you mis-specify the values of alpha_y and beta_y?
# 2) See if you can add in uncertainties to the reconstructed climates in the real example. Why are these uncertainties smaller than when extrapolate using a random walk?
# 3) (harder) One problem with the models above is the parameters of the state equation are assumed known. How might we estimate these parameters with unceratinty and then include it in the output?
