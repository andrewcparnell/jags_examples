# Header ------------------------------------------------------------------

# Fitting a logistic regression in JAGS
# Andrew Parnell

# In this file we fit a Bayesian Generalised Linear Model (GLM) in the form
# of a logistic regression.

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
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

# Likelihood
# y_t ~ Binomial(K,p_t), or Binomial(1,p_t) if binary
# logit(p_t) = alpha + beta_1 * x_1[t] + beta_2 * x_2[t]
# where logit(p_i) = log( p_t / (1 - p_t ))
# Note that p_t has to be between 0 and 1, but logit(p_t) has no limits

# Priors - all vague
# alpha ~ normal(0,100)
# beta_1 ~ normal(0,100)
# beta_2 ~ normal(0,100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T <- 100
set.seed(123)
x_1 <- sort(runif(T, 0, 10))
x_2 <- sort(runif(T, 0, 10))
alpha <- 1
beta_1 <- 0.2
beta_2 <- -0.5
logit_p <- alpha + beta_1 * x_1 + beta_2 * x_2
p <- inv.logit(logit_p)
y <- rbinom(T, 1, p)

# Have a quick look at the effect of x_1 and x_2 on y
plot(x_1, y)
plot(x_2, y) # Clearly when x is high y tends to be 0

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (t in 1:T) {
    y[t] ~ dbin(p[t], K)
    logit(p[t]) <- alpha + beta_1 * x_1[t] + beta_2 * x_2[t]
  }

  # Priors
  alpha ~ dnorm(0.0,0.01)
  beta_1 ~ dnorm(0.0,0.01)
  beta_2 ~ dnorm(0.0,0.01)
}
"

# Set up the data
model_data <- list(T = T, y = y, x_1 = x_1, x_2 = x_2, K = 1)

# Choose the parameters to watch
model_parameters <- c("alpha", "beta_1", "beta_2")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4,
  n.iter = 1000,
  n.burnin = 200,
  n.thin = 2
)

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
traceplot(model_run)

# Create a plot of the posterior mean regression line
post <- print(model_run)
alpha_mean <- post$mean$alpha
beta_1_mean <- post$mean$beta_1
beta_2_mean <- post$mean$beta_2

# As we have two explanatory variables I'm going to create two plots
# holding one of the variables fixed whilst varying the other
par(mfrow = c(2, 1))
plot(x_1, y)
lines(x_1,
  inv.logit(alpha_mean + beta_1_mean * x_1 + beta_2_mean * mean(x_2)),
  col = "red"
)
plot(x_2, y)
lines(x_2,
  inv.logit(alpha_mean + beta_1_mean * mean(x_1) + beta_2_mean * x_2),
  col = "red"
)

# Line for x_1 should be increasing with x_1, and vice versa with x_2

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory

# Adapted data from Royla and Dorazio (Chapter 2)
# Moth mortality data
T <- 12
K <- 20
y <- c(1, 4, 9, 13, 18, 20, 0, 2, 6, 10, 12, 16)
sex <- c(rep("male", 6), rep("female", 6))
dose <- rep(0:5, 2)
sexcode <- as.integer(sex == "male")
# The key questions is: what are the effects of dose and sex?

# Set up the data
real_data <- list(T = T, K = K, y = y, x_1 = sexcode, x_2 = dose)

# Run the mdoel
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

# Create same plot as before (only for does though)
post <- print(real_data_run)
alpha_mean <- post$mean$alpha
beta_1_mean <- post$mean$beta_1
beta_2_mean <- post$mean$beta_2

# Look at effect of sex - quantified by beta_1
hist(real_data_run$BUGSoutput$sims.list$beta_1, breaks = 30)
# Seems positive - males more likely to die

# What about effect of dose?
o <- order(real_data$x_2)
par(mfrow = c(1, 1)) # Reset plots
with(real_data, plot(x_2, y, pch = sexcode)) # Data
# Males
with(
  real_data,
  lines(x_2[o],
    K * inv.logit(alpha_mean + beta_1_mean * 1 + beta_2_mean * x_2[o]),
    col = "red"
  )
)
# Females
with(
  real_data,
  lines(x_2[o],
    K * inv.logit(alpha_mean + beta_1_mean * 0 + beta_2_mean * x_2[o]),
    col = "blue"
  )
)

# Legend
legend("topleft",
  legend = c("Males", "Females"),
  lty = 1,
  col = c("red", "blue")
)

# Other tasks -------------------------------------------------------------

# 1) See if there is an interaction between sex and dose in the above example. To do this add an extra term in the model beta_3 * x_1[i] * x_2[i]. Don't forget to include beta_3 in your parameters to watch vector. Is beta_3 precisely estimated in the posterior?
# 2) It almost always the case that the death rate increases with dose. Try changing the prior distribution on beta_2 to reflect the fact that the parameter should be positive. How much effect does this have on the posterior ditribution of beta_2?
# 3) (Harder) A common task is to estimate the LD-50, the dose at which 50% of the animals have died. See if you can estimate the LD-50 for males and females with uncertainty. (hint: either estimate a 95% CI for the LD50 for each group or, better yet, produce a full posterior distribution)
