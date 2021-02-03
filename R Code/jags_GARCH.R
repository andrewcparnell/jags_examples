# Header ------------------------------------------------------------------

# Generalised AutoRegressive Conditional Heteroskesticity (ARCH) models
# Andrew Parnell

# A GARCH model is just like an ARCH model but with an extra term for the previous variance. This code just fits a GARCH(1,1) model

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_t = response variable at time t=1,...,T
# alpha = overall mean
# sigma_t = residual standard deviation at time t
# gamma_1 = mean of variance term
# gamma_2 = AR component of variance
# Likelihood (see ARCH code also):
# y_t ~ N(mu, sigma_t^2)
# sigma_t^2 = gamma_1 + gamma_2 * (y_{t-1} - mu)^2 + gamma_3 * sigma_{t-1}^2

# Priors
# gamma_1 ~ unif(0,10) - needs to be positive
# gamma_2 ~ unif(0,1)
# gamma_3 ~ unif(0,1) - Be careful with these, as strictly gamma_2 + gamma_3 <1
# alpha ~ N(0,100) - vague

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T <- 100
alpha <- 1
gamma_1 <- 1
gamma_2 <- 0.4
gamma_3 <- 0.2
sigma <- y <- rep(NA, length = T)
set.seed(123)
sigma[1] <- runif(1)
y[1] <- 0
for (t in 2:T) {
  sigma[t] <- sqrt(gamma_1 + gamma_2 * (y[t - 1] - alpha)^2 + gamma_3 * sigma[t - 1]^2)
  y[t] <- rnorm(1, mean = alpha, sd = sigma[t])
}
plot(1:T, y, type = "l")

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (t in 1:T) {
    y[t] ~ dnorm(alpha, tau[t])
    tau[t] <- 1/pow(sigma[t], 2)
  }
  sigma[1] ~ dunif(0,10)
  for(t in 2:T) {
    sigma[t] <- sqrt( gamma_1 + gamma_2 * pow(y[t-1] - alpha, 2) + gamma_3 * pow(sigma[t-1], 2) )
  }

  # Priors
  alpha ~ dnorm(0.0, 0.01)
  gamma_1 ~ dunif(0, 10)
  gamma_2 ~ dunif(0, 1)
  gamma_3 ~ dunif(0, 1)
}
"

# Set up the data
model_data <- list(T = T, y = y)

# Choose the parameters to watch
model_parameters <- c("gamma_1", "gamma_2", "gamma_3", "alpha")

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

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
plot(model_run)
print(model_run)

# Real example ------------------------------------------------------------

# Run the GARCH(1,1) model on the ice core data set
ice <- read.csv("https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/GISP2_20yr.csv")
head(ice)
with(ice, plot(Age, Del.18O, type = "l"))
# Try plots of differences
with(ice, plot(Age[-1], diff(Del.18O, differences = 1), type = "l"))
with(ice, plot(Age[-(1:2)], diff(Del.18O, differences = 2), type = "l"))

# Try this on the last 30k years
ice2 <- subset(ice, Age >= 10000 & Age <= 25000)
table(diff(ice2$Age))
with(ice2, plot(Age[-1], diff(Del.18O), type = "l"))

# Set up the data
real_data <- with(
  ice2,
  list(T = nrow(ice2) - 1, y = diff(Del.18O))
)

# Save the sigma's the most interesting part!
model_parameters <- c("sigma", "alpha", "gamma_1", "gamma_2", "gamma_3")

# Run the model - requires longer to converge
real_data_run <- jags(
  data = real_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4,
  n.iter = 1000,
  n.burnin = 200,
  n.thin = 2
)

print(real_data_run)

# Have a look at the ARCH parameters;
par(mfrow = c(1, 3))
hist(real_data_run$BUGSoutput$sims.list$gamma_1, breaks = 30)
hist(real_data_run$BUGSoutput$sims.list$gamma_2, breaks = 30)
hist(real_data_run$BUGSoutput$sims.list$gamma_3, breaks = 30)
par(mfrow = c(1, 1))

# Plot the sigma outputs
sigma_med <- apply(real_data_run$BUGSoutput$sims.list$sigma, 2, "quantile", 0.5)
sigma_low <- apply(real_data_run$BUGSoutput$sims.list$sigma, 2, "quantile", 0.025)
sigma_high <- apply(real_data_run$BUGSoutput$sims.list$sigma, 2, "quantile", 0.975)

plot(ice2$Age[-1], sigma_med, type = "l", ylim = range(c(sigma_low, sigma_high)))
lines(ice2$Age[-1], sigma_low, lty = "dotted")
lines(ice2$Age[-1], sigma_high, lty = "dotted")


# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks
# 1) Try experimenting with the prior distributions of gamma_2 and gamma_3. Can you get any of the model runs to exceed the stationarity condition gamma_2+gamma_3 > 1. What happens to the plots? Try also simulating data with gamma_2 + gamma_3 > 1. What happens to the simulated example plots?
# 2) (Harder) This is a GARCH(1,1) model where the first argument refers to the previous residual and the second refers to the previous variance term. Try extending to more complicated GARCH models e.g. (1,2), (2,1) etc and look at the effect on the ice core data. (Hint: you migt be able to generalise using inprod as in the general AR(p) models)
