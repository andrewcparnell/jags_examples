# Header ------------------------------------------------------------------

# Fitting a linear regression in JAGS
# Andrew Parnell

# In this code we generate some data from a simple linear regression model and fit is using jags. We then intepret the output.

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

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
traceplot(model_run)

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

# Real example ------------------------------------------------------------

# Load in the Church and White global tide gauge data
sea_level <- read.csv("https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/church_and_white_global_tide_gauge.csv")
head(sea_level)
# We're ignore the errors and run the linear regression model.

# First plot
with(sea_level, plot(year_AD, sea_level_m))

# Run the jags code above

# Set up the data
real_data <- with(
  sea_level,
  list(
    n = nrow(sea_level),
    y = sea_level_m,
    x = year_AD
  )
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

# Plot output
print(real_data_run)

# Plot of posterior line
post <- print(real_data_run)
alpha_mean <- post$mean$alpha[1]
beta_mean <- post$mean$beta[1]

x <- sea_level$year_AD
with(sea_level, plot(year_AD, sea_level_m))
lines(x, alpha_mean + beta_mean * x, col = "red")
legend("topleft",
  legend = c("Data", "Posterior mean"),
  lty = c(-1, 1),
  pch = c(1, -1),
  col = c("black", "red")
)

# Other tasks -------------------------------------------------------------

# What about including the uncertainty on the line?
# We can do this using the posterior simulations of the slope/intercept
alpha_post <- real_data_run$BUGSoutput$sims.list$alpha
beta_post <- real_data_run$BUGSoutput$sims.list$beta
age_grid <- seq(1880, 2010, by = 10)
post_lines <- matrix(NA,
  ncol = length(age_grid),
  nrow = length(alpha_post)
)

# Now loop through each posterior value and get the fitted line
for (i in 1:nrow(post_lines)) {
  post_lines[i, ] <- alpha_post[i] + beta_post[i] * age_grid
}
# Now each row of post_lines contains a posterior sample of the fitted line

# We can plot them - here I'm plotting 50 random ones
with(sea_level, plot(year_AD, sea_level_m))
for (i in sample(1:nrow(post_lines), 50)) {
  lines(age_grid, post_lines[i, ], col = i)
}

# Or we can summarise them
post_lines_summary <- apply(post_lines,
  2,
  "quantile",
  probs = c(0.025, 0.5, 0.975)
)

with(sea_level, plot(year_AD, sea_level_m))
lines(age_grid, post_lines_summary[1, ], lty = 2, col = "red")
lines(age_grid, post_lines_summary[2, ], col = "red")
lines(age_grid, post_lines_summary[3, ], lty = 2, col = "red")
legend("topleft",
  legend = c("Posterior median", "Posterior 95% CI"),
  lty = c(1, 2),
  col = "red"
)

# Some other exercises
# 1) With the simulated data, experiment with changing the value of N when creating the data? What happens to the posterior distribution of the parameters as N gets bigger?
# 2) Try experimenting with the priors. Suppose that you *knew* that beta was negative and used the prior beta ~ dunif(-2,-1). What happens to the posterior mean lines?
# 3) (Harder) The sea level data has perhaps a slight quadratic trend. See if you can incorporate an extra parameter gamma multiplied by x^2. (Hint: in JAGS x^2 is written pow(x,2))
