# Header ------------------------------------------------------------------

# Fitting an over-dispersed (OD) logistic regression in JAGS
# Andrew Parnell

# In this file we fit a Bayesian Generalised Linear Model (GLM) in the form
# of an over-dispersed logistic regression.

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)
library(boot) # Package contains the logit transform

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_t = binomial (often binary) response variable for observation t=1,...,N
# x_{1t} = first explanatory variable for observation t
# x_{2t} = second " " " " " " " " "
# p_t = probability of y_t being 1 for observation t
# alpha = intercept term
# beta_1 = parameter value for explanatory variable 1
# beta_2 = parameter value for explanatory variable 2
# sigma = over-dispersion (OD) parameter

# Likelihood
# y_t ~ Binomial(K,p_t), or Binomial(1,p_t) if binary
# logit(p_t) ~ N(alpha + beta_1 * x_1[t] + beta_2 * x_2[t], sigma^2)
# where logit(p_i) = log( p_t / (1 - p_t ))
# Note that p_t has to be between 0 and 1, but logit(p_t) has no limits

# Priors - all quite tight as this model can be hard to identify
# alpha ~ normal(0,1)
# beta_1 ~ normal(0,1)
# beta_2 ~ normal(0,1)
# sigma ~ half-t(0, 1)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T = 100
K = 20
set.seed(123)
x_1 = sort(runif(T,0,10))
x_2 = sort(runif(T,0,10))
alpha = 1
beta_1 = 0.2
beta_2 = -0.5
sigma = 1
logit_p = rnorm(T, alpha + beta_1 * x_1 + beta_2 * x_2, sigma)
p = inv.logit(logit_p)
y = rbinom(T,K,p)

# Have a quick look at the effect of x_1 and x_2 on y
plot(x_1,y)
plot(x_2,y)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (t in 1:T) {
    y[t] ~ dbin(p[t], K)
    p[t] <- exp(logit_p[t])/(1 + exp(logit_p[t]))
    logit_p[t] ~ dnorm(alpha + beta_1 * x_1[t] + beta_2 * x_2[t], sigma^-2)
  }

  # Priors
  alpha ~ dnorm(0,1^-2)
  beta_1 ~ dnorm(0,1^-2)
  beta_2 ~ dnorm(0,1^-2)
  sigma ~ dt(0, 1^-2, 1)T(0,)
}
'

# Set up the data
model_data = list(T = T, y = y, x_1 = x_1, x_2 = x_2, K = 20)

# Choose the parameters to watch
model_parameters =  c("alpha", "beta_1", "beta_2", "sigma")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file = textConnection(model_code))

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
#traceplot(model_run)
