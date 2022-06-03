# Header ------------------------------------------------------------------

# Fitting a linear regression in JAGS with known measurement error
# Andrew Parnell

# In this code we generate some data from a simple linear regression model and fit is using jags. We then interpret the output.

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
# sigma_i = fixed extra error on y

# Likelihood:
# y[i] ~ N(alpha + beta * x[i], sigma^2 + sigma[i]^2)
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
sigma_i <- runif(n, 0, 2)
# Set the seed so this is repeatable
set.seed(123)
x <- sort(runif(n, 0, 10)) # Sort as it makes the plotted lines neater
y <- rnorm(n, mean = alpha + beta * x, sd = sqrt(sigma^2 + sigma_i^2))

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
    y[i] ~ dnorm(alpha + beta * x[i], total_var[i]^-1)
    total_var[i] <- sigma^2 + sigma_i[i]^2
  }

  # Priors
  alpha ~ dnorm(0, 100^-2)
  beta ~ dnorm(0, 100^-2)
  sigma ~ dunif(0, 10)
}
"

# Set up the data
model_data <- list(n = n, y = y, x = x, sigma_i = sigma_i)

# Choose the parameters to watch
model_parameters <- c("alpha", "beta", "sigma")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code))

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
# traceplot(model_run)

# Create a plot of the posterior mean regression line
post <- print(model_run)
alpha_mean <- post$mean$alpha[1]
beta_mean <- post$mean$beta[1]

plot(x, y)
lines(x, alpha_mean + beta_mean * x, col = "red")
lines(x, alpha + beta * x, col = "blue")
legend("topleft",
  legend = c("Truth", "Posterior mean"),
  lty = 1,
  col = c("blue", "red")
)
# Blue and red lines should be pretty close

