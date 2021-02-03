# Header ------------------------------------------------------------------

# Fitting a beta AR1 in JAGS
# Andrew Parnell and Ahmed Ali

# In this code we generate some data from a beta AR1 model and fit it using jags. We then intepret the output.

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)
library(boot)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_t = repsonse variable for observation t=1,..,N - should be in the range (0, 1)
# alpha, beta = intercept and slope parameters to be estimated
# phi - scale/variance(?) of beta distribution

# Likelihood:
# y_t ~ Beta(a[t], b[t])
# mu[t] = a[t]/(a[t] + b[t])
# a[t] = mu[t] * phi
# b[t] = (1 - mu[t]) * phi
# logit(mu[t]) = alpha + beta * mu[t-1]
# Prior
# alpha ~ N(0,100) - vague priors
# beta ~ N(0,100)
# phi ~ U(0, 100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T <- 100
alpha <- 0
beta <- 0.9
phi <- 5
sigma_mu <- 1
# Set the seed so this is repeatable
set.seed(124)
logit_mu <- rep(NA, length = T)
logit_mu[1] <- alpha
for (t in 2:T) {
  logit_mu[t] <- rnorm(
    1,
    alpha + beta * logit_mu[t - 1],
    sigma_mu
  )
}
mu <- inv.logit(logit_mu)
a <- mu * phi
b <- (1 - mu) * phi
y <- rbeta(T, a, b)

# Also creat a plot
plot(1:T, y)
lines(1:T, mu)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (t in 1:T) {
    y[t] ~ dbeta(a[t], b[t])
    a[t] <- mu[t] * phi
    b[t] <- (1 - mu[t]) * phi
    mu[t] <- ilogit(logit_mu[t])
  }
  logit_mu[1] ~ dnorm(alpha, 1)
  for(t in 2:T) {
    logit_mu[t] ~ dnorm(alpha + beta * logit_mu[t-1], sigma_mu^-2)
  }

  # Priors
  alpha ~ dnorm(0, 10^-2)
  beta ~ dnorm(0, 10^-2)
  phi ~ dunif(0, 10)
  sigma_mu ~ dunif(0, 10)
}
"

# Set up the data
model_data <- list(T = T, y = y)

# Choose the parameters to watch
model_parameters <- c("alpha", "beta", "phi", "sigma_mu", "mu")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code)
)
stop()
# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)

# Create a plot of the posterior mean regression line
post <- print(model_run)
mu_mean <- post$mean$mu

plot(1:T, y)
lines(1:T, mu_mean, col = "red")
lines(1:T, mu, col = "blue")
legend("topleft",
  legend = c("Truth", "Posterior mean"),
  lty = 1,
  col = c("blue", "red")
)
# Blue and red lines should be pretty close

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory
# Load in
library(datasets)
head(attenu)

# Set up the data
acc <- with(attenu, list(
  y = attenu$accel,
  T = nrow(attenu)
))
# Plot
plot(attenu$dist, attenu$accel,
  main = "Acceleration vs Distance",
  xlab = "Distance", ylab = "Accleration"
)

# Set up jags model
jags_model <- jags(acc,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4,
  n.iter = 1000,
  n.burnin = 200,
  n.thin = 2
)
# Plot the jags output
print(jags_model)

# Plot of posterior line
post <- print(jags_model)
alpha_mean <- post$mean$alpha
beta_mean <- post$mean$beta
mu_mean <- post$mean$mu
