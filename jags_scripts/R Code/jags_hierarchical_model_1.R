# Header ------------------------------------------------------------------

# Fitting a hierarchical linear model in JAGS
# Andrew Parnell

# In this code we generate some data from a single level hierarchical model (equivalently a random effects model) and fit it using JAGS. We then interpret the output

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_{ij} = repsonse variable for observation i=1,..,n_j in group j = 1,..,M.
# N = total number of observation = sum_j n_j
# alpha = overall mean parameter
# b_j = random effect for group j
# sigma = residual standard deviation
# sigma_b = standard deviation between groups

# Likelihood:
# y_{ij} ~ N(alpha + b_j, sigma^2)
# Prior
# alpha ~ N(0, 100) - vague priors
# b_j ~ N(0, sigma_b^2)
# sigma ~ half-cauchy(0, 10)
# sigma_b ~ half-cauchy(0, 10)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
M = 5 # Number of groups
alpha = 2
sigma = 1
sigma_b = 3
# Set the seed so this is repeatable
set.seed(123)
nj = sample(10:20, M, replace = TRUE) # Set the number of obs in each group between 5 and 10
N = sum(nj)
b = rnorm(M, 0, sigma_b)
group = rep(1:M, times = nj)
y = rnorm(N, mean = alpha + b[group], sd = sigma)

# Also creat a plot
boxplot(y ~ group)
points(1:M, alpha + b, col = 'red')

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data

model_code = '
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dnorm(alpha + b[group[i]], sigma^-2)
  }


  # Priors
  alpha ~ dnorm(0, 100^-2)
  for (j in 1:M) {
    b[j] ~ dnorm(0, sigma_b^-2)
  }
  sigma ~ dt(0, 10^-2, 1)T(0,)
  sigma_b ~ dt(0, 10^-2, 1)T(0,)
}
'

# Set up the data
model_data = list(N = N, y = y, M = M, group = group)

# Choose the parameters to watch
model_parameters =  c("alpha", "b", "sigma", "sigma_b")

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
alpha_mean = as.numeric(post$mean$alpha) # Need the as.numeric otherwise it stores it as a weird 1D object
b_mean = post$mean$b

boxplot(y ~ group)
points(1:M, alpha + b, col = 'red', pch = 19)
points(1:M, alpha_mean + b_mean, col = 'blue', pch = 19)
# Blue (true) and red (predicted) points should be pretty close

# Real example ------------------------------------------------------------
