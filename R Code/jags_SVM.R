# Header ------------------------------------------------------------------

# Stochastic volatility models in jags
# Andrew Parnell

# Stochastic volalility models (SVMs) are similar to ARCH/GARCH models in that they model changes in the variance. However SVMs give
# a probability distribution to the variances via a link function.

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)


# Maths -------------------------------------------------------------------

# Notation:
# y_t = response variable for discrete time t=1,...,T
# alpha = parameter to correct for mean of process
# h_t = transformed volatility parameters
# mu = mean of transformed volailities
# phi = AR parameter for transformed volatilities
# sigma = residual standard deviation for transformed volatilities

# Likelihood:
# y_t ~ normal( alpha, exp( h_t ) )
# h_t ~ normal( mu + phi * ( h_{t-1} - mu), sigma^2)

# Priors:
# alpha ~ normal(0, 100)
# mu ~ normal(0, 100)
# phi ~ uniform(-1, 1)
# sigma ~ uniform(0, 100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T <- 100
alpha <- -0.5
sigma <- 1
mu <- 0.5
phi <- 0.7
set.seed(123)
h <- rep(NA, length = T)
h[1] <- mu
for (t in 2:T) h[t] <- rnorm(1, mean = mu + phi * (h[t - 1] - mu), sd = sigma)
y <- rnorm(T, mean = alpha, sd = exp(h / 2))
par(mfrow = c(2, 1))
plot(1:T, y, type = "l")
plot(1:T, exp(h / 2), type = "l")
par(mfrow = c(1, 1))

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (t in 1:T) {
    y[t] ~ dnorm(alpha, tau_h[t])
    tau_h[t] <- 1/exp(h[t])
  }
  h[1] <- mu
  for(t in 2:T) {
    h[t] ~ dnorm(mu + phi * (h[t-1] - mu), tau)
  }

  # Priors
  alpha ~ dnorm(0, 0.01)
  mu ~ dnorm(0, 0.01)
  phi ~ dunif(-1, 1)
  tau <- 1/pow(sigma, 2)
  sigma ~ dunif(0,100)
}
"

# Set up the data
model_data <- list(T = T, y = y)

# Choose the parameters to watch
model_parameters <- c("alpha", "mu", "phi", "sigma")

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

# Fit to the ice core data like the ARCH/GARCH models
ice <- read.csv("https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/GISP2_20yr.csv")
head(ice)
with(ice, plot(Age, Del.18O, type = "l"))
# Try plots of differences
with(ice, plot(Age[-1], diff(Del.18O, differences = 1), type = "l"))

# Try this on the last 30k years
ice2 <- subset(ice, Age >= 10000 & Age <= 25000)
table(diff(ice2$Age))
with(ice2, plot(Age[-1], diff(Del.18O), type = "l"))

# Set up the data
real_data <- with(
  ice2,
  list(T = nrow(ice2) - 1, y = diff(Del.18O))
)

# Choose the parameters to watch
model_parameters <- c("alpha", "mu", "phi", "sigma", "h")

# Run the model - requires longer to converge
real_data_run <- jags(
  data = real_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4,
  n.iter = 10000,
  n.burnin = 2000,
  n.thin = 8
)

print(real_data_run)

# Have a look at the ARCH parameters;
par(mfrow = c(2, 2))
hist(real_data_run$BUGSoutput$sims.list$alpha, breaks = 30)
hist(real_data_run$BUGSoutput$sims.list$mu, breaks = 30)
hist(real_data_run$BUGSoutput$sims.list$phi, breaks = 30)
hist(real_data_run$BUGSoutput$sims.list$sigma, breaks = 30)
par(mfrow = c(1, 1))

# Plot the h outputs
h_med <- exp(apply(real_data_run$BUGSoutput$sims.list$h, 2, "quantile", 0.5) / 2)
h_low <- exp(apply(real_data_run$BUGSoutput$sims.list$h, 2, "quantile", 0.025) / 2)
h_high <- exp(apply(real_data_run$BUGSoutput$sims.list$h, 2, "quantile", 0.975) / 2)

plot(ice2$Age[-1], h_med, type = "l", ylim = range(c(h_low[-1], h_high[-1])))
lines(ice2$Age[-1], h_low, lty = "dotted")
lines(ice2$Age[-1], h_high, lty = "dotted")

# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks
# 1) Try re-running the above model on the raw ice core data series rather than the differences. Do the model predictions of h change much?
# 2) (harder) Try adding a random walk term in to mean of the stochastic volatility model. Does this fit the data better? (How can you tell?) Do the predictions of h change much?
# 3) (even harder) See if you can use some of the ideas from ARCH/GARCH models in the mean of the process for h. For example, does including the previous lag of the data in the process for h improve the model?
