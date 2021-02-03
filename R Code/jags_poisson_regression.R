# Header ------------------------------------------------------------------

# Fitting a poisson regression in JAGS
# Bruna Wundervald

# In this file we fit a Bayesian Generalised Linear Model (GLM) in the form
# of a poisson regression.

# Loading packages
library(R2jags)
library(tidyverse)
library(patchwork) # devtools::install_github("thomasp85/patchwork")

# Description of the Bayesian Poisson model
# Notation:
# y_i = poisson response variable for observation t = 1,...,N
# x_{1i} = first explanatory variable for observation t
# x_{2i} = second " " " " " " " " "
# lambda_i = the rate of occurence for each random variable y_t
# alpha = intercept term
# beta_1 = parameter value for explanatory variable 1
# beta_2 = parameter value for explanatory variable 2

# Likelihood
# y_i ~ Poisson(lambda_i)
# log(lambda_i) = alpha + beta_1 * x_1[i] + beta_2 * x_2[i]
# lambda_i must be bigger than 0, but log(lambda_i) has no limits

# Priors - all vague
# alpha ~ normal(0, 100)
# beta_1 ~ normal(0, 100)
# beta_2 ~ normal(0, 100)


# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T <- 1000
set.seed(123)
x_1 <- sort(runif(T, 0, 5))
x_2 <- sort(runif(T, 0, 5))
alpha <- 1
beta_1 <- 1.2
beta_2 <- -0.3
mu <- alpha + beta_1 * x_1 + beta_2 * x_2
lambda <- exp(mu)
y <- rpois(n = T, lambda = lambda)


df <- data.frame(y, x_1, x_2)

# Have a quick look at the effect of x_1 and x_2 on y

p1 <- df %>%
  ggplot(aes(x_1, y)) +
  geom_point(colour = "orange") +
  labs(title = "Y versus the explanatory variables and 
Y versus the exponential of explanatory variables") +
  theme_bw()

p2 <- df %>%
  ggplot(aes(exp(x_1), y)) +
  geom_point(colour = "orange") +
  theme_bw()


p3 <- df %>%
  ggplot(aes(x_2, y)) +
  geom_point(colour = "orange") +
  theme_bw()

p4 <- df %>%
  ggplot(aes(exp(x_2), y)) +
  geom_point(colour = "orange") +
  theme_bw()


p1 + p2 + p3 + p4 + plot_layout(ncol = 2)


# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (i in 1:T) {
    y[i] ~ dpois(p[i])
    log(p[i]) <- alpha + beta_1 * x_1[i] + beta_2 * x_2[i]
  }
  # Priors
  alpha ~ dnorm(0.0, 0.01)
  beta_1 ~ dnorm(0.0, 0.01)
  beta_2 ~ dnorm(0.0, 0.01)
}
"

# Set up the data
model_data <- list(T = T, y = y, x_1 = x_1, x_2 = x_2)

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

# Creating plots of the posterior regression line
post <- print(model_run)
alpha_mean <- post$mean$alpha
beta_1_mean <- post$mean$beta_1
beta_2_mean <- post$mean$beta_2

df <- df %>%
  mutate(pred = exp(alpha_mean + beta_1_mean * x_1 + beta_2_mean * x_2))

p12 <- p1 +
  geom_line(data = df, aes(y = pred), colour = "royalblue") +
  labs(title = "Y versus the explanatory variables with predicted line")

p32 <- p3 +
  geom_line(data = df, aes(y = pred), colour = "royalblue")

p12 + p32 + plot_layout(nrow = 1)

# Real example ------------------------------------------------------------
# install_github(repo = "labestData", username = "pet-estatistica",
#               ref = "master", build_vignettes = TRUE)

df <- labestData::PaulaEx4.6.7

# This data is related to pieces of fabric. The response variable
# is the count of failures in each piece and the covariable is
# the length of the piece in meters


# Jags code to fit the model to the real data
model_code <- "
model
{
  # Likelihood
  for (i in 1:T) {
    y[i] ~ dpois(p[i])
    log(p[i]) <- alpha + beta_1 * x_1[i]
  }
  # Priors
  alpha ~ dnorm(0.0, 0.01)
  beta_1 ~ dnorm(0.0, 0.01)
  beta_2 ~ dnorm(0.0, 0.01)
}
"

# Set up the data
model_data <- list(T = dim(df)[1], y = df$nfalhas, x_1 = df$comp)

# Choose the parameters to watch
model_parameters <- c("alpha", "beta_1")

# Run the model
real_data_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 4,
  n.iter = 1000,
  n.burnin = 200,
  n.thin = 2
)

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(real_data_run)
print(real_data_run)
traceplot(real_data_run)


# Creating plots of the posterior regression line
post <- print(real_data_run)
alpha_mean <- post$mean$alpha
beta_1_mean <- post$mean$beta_1

df <- df %>%
  mutate(pred = exp(alpha_mean + beta_1_mean * comp))


df %>%
  ggplot(aes(comp, nfalhas)) +
  geom_point(colour = "orange") +
  geom_line(data = df, aes(y = pred), colour = "royalblue") +
  labs(title = "Y versus the explanatory variable with predicted line") +
  theme_bw()
