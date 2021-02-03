# Header ------------------------------------------------------------------

# Fitting a logistic regression in JAGS with measurement error
# Andrew Parnell

# In this file we fit a Bayesian Generalised Linear Model (GLM) in the form
# of a logistic regression with measurement error in the covariates

# Some boiler plate code to clear the workspace, and load in required packages
rm(list = ls()) # Clear the workspace
library(R2jags)
library(boot) # Package contains the logit transform

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_i = binomial (often binary) response variable for observation i=1,...,N
# x_{1i} = first explanatory variable for observation i
# x_{2i} = second " " " " " " " " "
# mu_{1i} = known measurement error mean for explanatory variable 1
# mu_{2i} = known measurement error mean for explanatory variable 2
# sig_{1i} = known measurement error sd for explanatory variable 1
# sig_{2i} = known measurement error sd for explanatory variable 2
# p_i = probability of y_i being 1 for observation i
# alpha = intercept term
# beta_1 = parameter value for explanatory variable 1
# beta_2 = parameter value for explanatory variable 2

# Likelihood
# y_i ~ Binomial(K,p_i), or Binomial(1,p_i) if binary
# logit(p_i) = alpha + beta_1 * x_1[i] + beta_2 * x_2[i]
# x_1i ~ dnorm(mu_1i, sigma_1i^2) # Note mu and sigma are DATA - you can make them parameters if required
# x_2i ~ dnorm(mu_2i, sigma_2i^2)
# where logit(p_i) = log( p_i / (1 - p_i ))

# Priors - all vague
# alpha ~ normal(0,100)
# beta_1 ~ normal(0,100)
# beta_2 ~ normal(0,100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
N <- 100
set.seed(123)
mu_1 <- sort(runif(N, 0, 10))
mu_2 <- sort(runif(N, 0, 10))
sigma_1 <- runif(N, 0.1, 0.3)
sigma_2 <- runif(N, 0.1, 0.3)
x_1 <- rnorm(N, mu_1, sigma_1)
x_2 <- rnorm(N, mu_2, sigma_2)
alpha <- 1
beta_1 <- 0.2
beta_2 <- -0.5
logit_p <- alpha + beta_1 * mu_1 + beta_2 * mu_2
p <- inv.logit(logit_p)
y <- rbinom(N, 1, p)

# Have a quick look at the effect of x_1 and x_2 on y
plot(x_1, y)
plot(x_2, y) # Clearly when x is high y tends to be 0

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dbin(p[i], K)
    logit(p[i]) <- alpha + beta_1 * x_1[i] + beta_2 * x_2[i]
    x_1[i] ~ dnorm(mu_1[i], sigma_1[i]^-2)
    x_2[i] ~ dnorm(mu_2[i], sigma_2[i]^-2)
  }

  # Priors
  alpha ~ dnorm(0, 10^-2)
  beta_1 ~ dnorm(0, 10^-2)
  beta_2 ~ dnorm(0, 10^-2)
}
"

# Set up the data
model_data <- list(
  N = N, y = y, mu_1 = mu_1, sigma_1 = sigma_1,
  mu_2 = mu_2, sigma_2 = sigma_2, K = 1
)

# Choose the parameters to watch
model_parameters <- c("alpha", "beta_1", "beta_2", "x_1", "x_2")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code)
)

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)

# Create a plot of the posterior mean regression line
post <- print(model_run)
alpha_mean <- post$mean$alpha[1]
beta_1_mean <- post$mean$beta_1[1]
beta_2_mean <- post$mean$beta_2[1]
x_1_mean <- post$mean$x_1
x_2_mean <- post$mean$x_2

# As we have two explanatory variables I'm going to create two plots
# holding one of the variables fixed whilst varying the other
par(mfrow = c(2, 1))
plot(x_1_mean, y)
points(x_1_mean,
  inv.logit(alpha_mean + beta_1_mean * x_1_mean + beta_2_mean * mean(x_2_mean)),
  col = "red"
)
plot(x_2_mean, y)
points(x_2,
  inv.logit(alpha_mean + beta_1_mean * mean(x_1_mean) + beta_2_mean * x_2_mean),
  col = "red"
)
par(mfrow = c(1, 1))

# See how well it estimated x_1 and x_2
plot(x_1, x_1_mean) # A Good fit
plot(x_2, x_2_mean)
