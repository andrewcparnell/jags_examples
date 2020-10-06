# Header ------------------------------------------------------------------

# Createing a posterior predictive distribution in JAGS
# This works for any statistical model, but I'm going to fit it just for a linear regressio model
# Andrew Parnell

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_t = repsonse variable for observation t=1,..,N
# x_t = explanatory variable for obs t
# alpha, beta = intercept and slope parameters to be estimated
# sigma = residual standard deviation

# Likelihood:
# y_t ~ N(alpha + beta * x[i], sigma^2)
# Prior
# alpha ~ N(0,100) - vague priors
# beta ~ N(0,100)
# sigma ~ U(0,10)

# Posterior predictive will be a simulation from the likelihood probability distribution for the current values of the parameters
# y_pred[i] ~ N(alpha + beta*x[i], sigma^2)
# y_pred will be treated as a set of parameters

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
N = 100
alpha = 2
beta = 3
sigma = 1
# Set the seed so this is repeatable
set.seed(123)
x = sort(runif(N, 0, 10)) # Sort as it makes the plotted lines neater
y = rnorm(N, mean = alpha + beta * x, sd = sigma)

# Also creat a plot
plot(x, y)
lines(x, alpha + beta * x)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data

model_code = '
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dnorm(mu[i], sigma^-2)
    mu[i] <- alpha + beta * x[i]
    y_pred[i] ~ dnorm(mu[i], sigma^-2) # This is the key extra line to include to get the posterior predictive
  }

  # Priors
  alpha ~ dnorm(0, 100^-2)
  beta ~ dnorm(0, 100^-2)
  sigma ~ dunif(0, 10)
}
'

# Set up the data
model_data = list(N = N, y = y, x = x)

# Choose the parameters to watch
model_parameters =  c("y_pred")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code))

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)

# Check the posterior predictive

# Get the posterior samples of y_pred
y_pred = model_run$BUGSoutput$sims.list$y_pred

# Create 50% intervals (change to e.g. 0.025, 0.5, 0.975, for 95% intervals)
y_pred_quant = apply(y_pred, 2, quantile, probs = c(0.25, 0.5, 0.75))

# Create a plot of the true values against the posterior predicted values
plot(y, y_pred_quant[2,], pch = 19)
for(i in 1:ncol(y_pred_quant)) {
  lines(c(y[i], y[i]), c(y_pred_quant[1,i], y_pred_quant[3,i]))
}
abline(a = 0, b = 1)
# Points should be close to the line

# Real example ------------------------------------------------------------

# Load in the Church and White global tide gauge data
sea_level = read.csv('https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/church_and_white_global_tide_gauge.csv')
head(sea_level)
# We're ignore the errors and run the linear regression model.

# First plot
with(sea_level,plot(year_AD,sea_level_m))

# Run the jags code above

# Set up the data
real_data = with(sea_level,
                  list(N = nrow(sea_level),
                       y = sea_level_m,
                       x = year_AD))

# Run the model
real_data_run = jags(data = real_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code))

# Plot output
plot(real_data_run)

# See the jags_linear_regression example for a plot of the posterior fitted line

# Plot of posterior predicted values
y_pred = real_data_run$BUGSoutput$sims.list$y_pred

# Create 50% intervals (change to e.g. 0.025, 0.5, 0.975, for 95% intervals)
y_pred_quant = apply(y_pred, 2, quantile, probs = c(0.25, 0.5, 0.75))

# Create a plot of the true values against the posterior predicted values
plot(real_data$y, y_pred_quant[2,], pch = 19)
for(i in 1:ncol(y_pred_quant)) {
  lines(c(real_data$y[i], real_data$y[i]), c(y_pred_quant[1,i], y_pred_quant[3,i]))
}
abline(a = 0, b = 1)
# A little bit curved - perhaps a quadratic or non-parametric model might perform better?