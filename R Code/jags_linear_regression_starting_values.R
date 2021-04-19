# Header ------------------------------------------------------------------

# Fitting a linear regression in JAGS. In this version we show how to fit the model sequentially.
# Andrew Parnell

# We first fit the model for 100 iterations. Then use the final values from that model as the starting values for the next model. This technique is useful when the model is very slow to run so we want to monitor progress

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_i = repsonse variable for observation t=i,..,N
# x_i = explanatory variable for obs i
# alpha, beta = intercept and slope parameters to be estimated
# sigma = residual standard deviation

# Likelihood:
# y[i] ~ N(alpha + beta * x[i], sigma^2)
# Prior
# alpha ~ N(0,100) - vague priors
# beta ~ N(0,100)
# sigma ~ U(0,10)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
n <- 100
alpha <- 2
beta <- 3
sigma <- 1
# Set the seed so this is repeatable
set.seed(123)
x <- sort(runif(n, 0, 10)) # Sort as it makes the plotted lines neater
y <- rnorm(n, mean = alpha + beta * x, sd = sigma)

# Also creat a plot
plot(x, y)
lines(x, alpha + beta * x)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data

model_code <- "
model
{
  # Likelihood
  for (i in 1:n) {
    y[i] ~ dnorm(alpha + beta * x[i], sigma^-2)
  }

  # Priors
  alpha ~ dnorm(0, 100^-2)
  beta ~ dnorm(0, 100^-2)
  sigma ~ dunif(0, 10)
}
"

# Set up the data
model_data <- list(n = n, y = y, x = x)

# Choose the parameters to watch
model_parameters <- c("alpha", "beta", "sigma")

# Run the model
model_run_1 <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4, # Number of different starting positions
  n.iter = 1000, # Number of iterations
  n.burnin = 0, # Number of iterations to remove at start
  n.thin = 1
) # Amount of thinning

# Now run the model again but using the final values as the starting values for the new run
new_inits <- vector('list', length = 4)
for(i in 1:length(new_inits)) {
  new_inits[[i]] <- as.list(tail(model_run_1$BUGSoutput$sims.array[,i,], 1))
  names(new_inits[[i]]) <- names(model_run_1$BUGSoutput$sims.list)
}

model_run_2 <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  inits = new_inits,
  n.chains = 4, # Number of different starting positions
  n.iter = 1000, # Number of iterations
  n.burnin = 0, # Number of iterations to remove at start
  n.thin = 1
) # Amount of thinning


# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run_2)
print(model_run_2)
traceplot(model_run_2)

