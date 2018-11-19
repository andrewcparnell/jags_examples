# Header ------------------------------------------------------------------

# Fitting a beta linear regression in JAGS
# Andrew Parnell

# In this code we generate some data from a beta linear regression model and fit is using jags. We then intepret the output.

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)
library(boot)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_t = repsonse variable for observation t=1,..,N - should be in the range (0, 1)
# x_t = explanatory variable for obs t
# alpha, beta = intercept and slope parameters to be estimated
# sigma = residual standard deviation

# Likelihood:
# y_t ~ Beta(a[t], b[t])
# mu[t] = a[t]/(a[t] + b[t])
# a[t] = mu[t] * phi
# b[t] = (1 - mu[t]) * phi
# logit(mu[t]) = alpha + beta * x[t]
# Prior
# alpha ~ N(0,100) - vague priors
# beta ~ N(0,100)
# phi ~ U(0, 100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T = 100
alpha = -1
beta = 0.2
phi = 5
# Set the seed so this is repeatable
set.seed(123)
x = sort(runif(T, 0, 10)) # Sort as it makes the plotted lines neater
logit_mu = alpha + beta * x
mu = inv.logit(logit_mu)
a = mu * phi
b = (1 - mu) * phi
y = rbeta(T, a, b)

# Also creat a plot
plot(x, y)
lines(x, mu)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data

model_code = '
model
{
  # Likelihood
  for (t in 1:T) {
    y[t] ~ dbeta(a[t], b[t])
    a[t] <- mu[t] * phi
    b[t] <- (1 - mu[t]) * phi
    logit(mu[t]) <- alpha + beta * x[t]
  }

  # Priors
  alpha ~ dnorm(0, 10^-2)
  beta ~ dnorm(0, 10^-2)
  phi ~ dunif(0, 10)
}
'

# Set up the data
model_data = list(T = T, y = y, x = x)

# Choose the parameters to watch
model_parameters =  c("alpha","beta","phi")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code))

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
traceplot(model_run)

# Create a plot of the posterior mean regression line
post = print(model_run)
alpha_mean = post$mean$alpha
beta_mean = post$mean$beta

plot(x, y)
lines(x, inv.logit(alpha_mean + beta_mean * x), col = 'red')
lines(x, inv.logit(alpha + beta * x), col = 'blue')
legend('topleft',
       legend = c('Truth', 'Posterior mean'),
       lty=1,
       col=c('blue','red'))
# Blue and red lines should be pretty close
