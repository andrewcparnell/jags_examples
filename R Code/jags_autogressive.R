# Header ------------------------------------------------------------------

# Autoregressive model of order p
# Andrew Parnell

# Some JAGS code to fit AR(1) and AR(p) models

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y(t) = response variable at time t, t = 1,...,T
# alpha = overall mean parameter
# phi = autocorrelation/autoregressive (AR) parameter
# phi_j = Some of the models below have multiple AR parameters, j = 1,..P
# sigma = residual standard deviation

# Likelihood
# For AR(1)
# y[t] ~ normal(alpha + phi * y[t-1], sigma^2)
# For AR(p)
# y[t] ~ normal(alpha + phi[1] * y[t-1] + ... + phi[p] * y[y-p], sigma^2)

# Priors
# alpha ~ dnorm(0,100)
# phi ~ dunif(-1,1) # If you want the process to be stable/stationary
# phi ~ dnorm(0,100) # If you're not fussed about stability
# sigma ~ dunif(0,100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
# First AR1
set.seed(123)
T <- 100
t_seq <- 1:T
sigma <- 1
alpha <- 1
phi <- 0.6
y <- rep(NA, T)
y[1] <- rnorm(1, 0, sigma)
for (t in 2:T) y[t] <- rnorm(1, alpha + phi * y[t - 1], sigma)
# plot
plot(t_seq, y, type = "l")

# Also simulate an AR(p) process
p <- 3
phi2 <- c(0.5, 0.1, -0.02)
y2 <- rep(NA, T)
y2[1:p] <- rnorm(p, 0, sigma)
for (t in (p + 1):T) y2[t] <- rnorm(1, alpha + sum(phi2 * y2[(t - 1):(t - p)]), sigma)
plot(t_seq, y2, type = "l")

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
# This code is for a general AR(p) model

model_code <- "
model
{
  # Likelihood
  for (t in (p+1):T) {
    y[t] ~ dnorm(mu[t], tau)
    mu[t] <- alpha + inprod(phi, y[(t-p):(t-1)])
  }

  # Priors
  alpha ~ dnorm(0.0,0.01)
  for (i in 1:p) {
    phi[i] ~ dnorm(0.0,0.01)
  }
  tau <- 1/pow(sigma,2) # Turn precision into standard deviation
  sigma ~ dunif(0.0,10.0)
}
"

# Set up the data
model_data <- list(T = T, y = y, p = 1)

# Choose the parameters to watch
model_parameters <- c("alpha", "phi", "sigma")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4, # Number of different starting positions
  n.iter = 1000, # Number of iterations
  n.burnin = 200, # Number of iterations to remove at start
  n.thin = 2
) # Amount of thinning

# Try the AR(p) version
model_data_2 <- list(T = T, y = y2, p = p)

model_run_2 <- jags(
  data = model_data_2,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4, # Number of different starting positions
  n.iter = 1000, # Number of iterations
  n.burnin = 200, # Number of iterations to remove at start
  n.thin = 2
) # Amount of thinning

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
print(model_run)
print(model_run_2) # Note: phi is correct but in the wrong order

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory
hadcrut <- read.csv("https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/hadcrut.csv")
head(hadcrut)
with(hadcrut, plot(Year, Anomaly, type = "l"))

# Look at the ACF/PACF
acf(hadcrut$Anomaly)
pacf(hadcrut$Anomaly)

# Set up the data
real_data <- with(
  hadcrut,
  list(
    T = nrow(hadcrut),
    y = hadcrut$Anomaly,
    p = 1
  )
)

# Run the model
real_data_run <- jags(
  data = real_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4,
  n.iter = 1000,
  n.burnin = 200,
  n.thin = 2
)

# Plot output
print(real_data_run) # Very high degree of autocorrelation

# Plot some of the fitted values (also known as one-step-ahead predictions)
post <- print(real_data_run)
alpha_mean <- post$mean$alpha
phi_mean <- post$mean$phi

# Create fitted values
fitted_values <- alpha_mean + phi_mean * real_data$y[1:(nrow(hadcrut) - 1)]

# Create fitted line
with(hadcrut, plot(Year, Anomaly, type = "l"))
with(hadcrut, lines(Year[2:nrow(hadcrut)], fitted_values, col = "red"))
# Why does this look strange?

# Create some predictions off into the future
T_future <- 2050
future_values <- rep(NA, T_future - max(hadcrut$Year))
future_values[1] <- alpha_mean + phi_mean * real_data$y[nrow(hadcrut)]
for (i in 2:length(future_values)) {
  future_values[i] <- alpha_mean + phi_mean * future_values[i - 1]
}

# Plot these all together
with(
  hadcrut,
  plot(Year,
    Anomaly,
    type = "l",
    xlim = c(min(hadcrut$Year), T_future),
    ylim = range(c(hadcrut$Anomaly, future_values))
  )
)
lines(((max(hadcrut$Year) + 1):T_future), future_values, col = "red")
# See - no global warming!

# Other tasks -------------------------------------------------------------

# 1) Try changing the values of phi2 for the simulated AR(p) model. What happens to the time series when some of these values get bigger?
# 2) Above we have only fitted the HadCrut data with an AR(1) model. You might like to try and fit it with AR(2), AR(3), etc models and see what happens to the fits
# 3) (Harder) See if you can create the fitted values by sampling from the posterior distribution of alpha and phi, and plotting an envelope/ensemble of lines, just like in the linear regression example
