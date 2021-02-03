# Header ------------------------------------------------------------------

# Autoregressive model with repeated measures of order 1
# Andrew Parnell

# Some JAGS code to fit a repeated measures AR(1)

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_i(t) = response variable at time t, t = 1,...,T
# alpha = overall mean parameter
# b_t = random effect for time t
# phi = autocorrelation/autoregressive (AR) parameter
# phi_j = Some of the models below have multiple AR parameters, j = 1,..P
# sigma_y = standard deviation for repeated measures
# sigma_b = residual standard deviation of time series

# Likelihood
# For AR(1)
# y[i,t] ~ normal(alpha + b[t], sigma_y^2)
# b_t ~ normal(phi * b[t-1], sigma_b^2)

# Priors
# alpha ~ dnorm(0,100)
# phi ~ dunif(-1,1) # If you want the process to be stable/stationary
# phi ~ dnorm(0,100) # If you're not fussed about stability
# sigma_y ~ dunif(0,100)
# sigma_b ~ dunif(0,100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
# First AR1
set.seed(123)
T <- 100
num_obs_per_time <- sample(1:5, T, replace = TRUE) # Num obs per time point
t_seq <- 1:T
t_seq_2 <- rep(t_seq, times = num_obs_per_time)
sigma_b <- 1
sigma_y <- 0.5
alpha <- 1
phi <- 0.6
b <- y <- rep(NA, T)
b[1] <- rnorm(1, 0, sigma_b)
for (t in 2:T) b[t] <- rnorm(1, phi * b[t - 1], sigma_b)
for (i in 1:length(t_seq_2)) {
  y[i] <- rnorm(1, mean = alpha + b[t_seq_2[i]], sigma_y)
}

# plot
plot(t_seq_2, y, type = "p")

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
# This code is for a general AR(p) model

model_code <- "
model
{
  # Likelihood
  for (i in 1:T_big) {
    y[i] ~ dnorm(mu[i], tau_y)
    mu[i] <- b[b_select[i]]
  }
  b[1] ~ dnorm(alpha, tau_b)
  for(t in 2:T) {
    b[t] ~ dnorm(alpha + phi * b[t-1], tau_b)
  }

  # Priors
  alpha ~ dnorm(0.0,0.01)
  phi ~ dunif(-1,1)
  tau_y <- 1/pow(sigma_y,2)
  sigma_y ~ dunif(0.0,10.0)
  tau_b <- 1/pow(sigma_b,2)
  sigma_b ~ dunif(0.0,10.0)
}
"

# Set up the data
model_data <- list(
  T = T,
  T_big = length(t_seq_2),
  b_select = t_seq_2,
  y = y
)

# Choose the parameters to watch
model_parameters <- c("alpha", "phi", "sigma_y", "sigma_b")

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

# Simulated results -------------------------------------------------------

print(model_run)
print(model_run)

# Fitted values -----------------------------------------------------------

# Choose the parameters to watch
model_parameters <- c("b")

# Run the model
model_run_fit <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4, # Number of different starting positions
  n.iter = 1000, # Number of iterations
  n.burnin = 200, # Number of iterations to remove at start
  n.thin = 2
) # Amount of thinning

# plot
plot(t_seq_2, y, type = "p")
lines(t_seq, model_run_fit$BUGSoutput$mean$b, col = "red")
