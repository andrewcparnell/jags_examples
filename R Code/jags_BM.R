# Header ------------------------------------------------------------------

# Simple Brownian motion in jags
# Andrew Parnell

# This model fit a simple Brownian motion model in JAGS, out first foray into proper continuous time series

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y(t) = response variable observated at continuous times t
# alpha = optional drift parameter
# sigma = standard deviation/volatility parameter

# Likelihood:
# Sometimes written as dy = alpha dt + sigma * dW(t), but more helpfully written as
# y(t) - y(t - s) ~ N(alpha * (t - s), s * sigma^2) where s is any positive value

# Priors:
# alpha ~ N(0, 100)
# sigma ~ uniform(0, 100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T <- 100
alpha <- 0
sigma <- 1
t <- sort(runif(T, 0, 1)) # Assume time runs from 0 to 1
y <- rep(NA, T)
y[1] <- 0
for (i in 2:T) y[i] <- y[i - 1] + rnorm(1, alpha * (t[i] - t[i - 1]), sigma * sqrt(t[i] - t[i - 1]))
plot(t, y, type = "l")

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (i in 2:T) {
  y[i] ~ dnorm( alpha * (t[i] - t[i-1]) + y[i-1], tau[i] )
  tau[i] <- 1/( pow(sigma,2) * (t[i] - t[i-1]) )
  }
  
  # Priors
  alpha ~ dnorm(0.0,0.01)
  sigma ~ dunif(0.0,10.0)
}
"

# Set up the data
model_data <- list(T = T, y = y, t = t)

# Choose the parameters to watch
model_parameters <- c("alpha", "sigma")

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
print(model_run)
plot(model_run)

# Real example ------------------------------------------------------------

# Let's now fit this to the entire ice core
ice <- read.csv("https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/GISP2_20yr.csv")
head(ice)
with(ice, plot(Age, Del.18O, type = "l"))

# Set up the data
real_data <- with(
  ice,
  list(y = Del.18O, T = nrow(ice), t = Age)
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

print(real_data_run)
plot(real_data_run)

# Note that with a small change to the data we can now get predicted
# ice core values for any time slot we would like - this is the same trick
# we used in the ARIMA and ARIMAX files
t_ideal <- seq(0 + 0.01, max(ice$Age) + 0.01, by = 100) # 100 year regular grid
# Note added on 0.01 to the above to stop there being some zero time differences
y_ideal <- rep(NA, length(t_ideal))
t_all <- c(ice$Age, t_ideal)
y_all <- c(ice$Del.18O, y_ideal)
o <- order(t_all)
t_all[o][1:10]
y_all[o][1:10]

# Create new data set
real_data_2 <- with(
  ice,
  list(y = y_all[o], T = length(y_all), t = t_all[o])
)

# Save all the values of y
model_parameters <- "y"

# Run the model - if the below is slow to run try reducing the time grid above
real_data_run_2 <- jags(
  data = real_data_2,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4,
  n.iter = 1000,
  n.burnin = 200,
  n.thin = 2
)

plot(real_data_run_2)

# Now create a plot of the gridded predicted values
pick_out <- which(is.na(real_data_2$y))
pred_y <- apply(real_data_run_2$BUGSoutput$sims.list$y[, pick_out], 2, "mean")

plot(t_ideal, pred_y, type = "l")
# We now have estimates based on an artbitrary 100 year grid!

# Other tasks -------------------------------------------------------------

# 1) Try simulating some data using non-zero values of alpha. How do the simulated time series change? Does the jags model estimate alpha well?
# 2) Try fiddling with the predicted grid in t_ideal. You might like to focus on producing plots for a particular time on an annual scale, or to extrapolate off into the past to see what happens
# 3) See if you can add in the 95% uncertainty bounds to the gridded prediction plot from above
