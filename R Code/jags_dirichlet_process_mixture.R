# Header ------------------------------------------------------------------

# Fitting a Dirichlet Process Mixture Model in JAGS
# Code mostly taken from here: https://www.jarad.me/courses/stat615/slides/Nonparametrics/nonparametrics.pdf
# Andrew Parnell

# In this file we fit a Dirichlet Process Gaussian Mixture model using two simple components

# Loading in required packages
library(R2jags)
library(tidyverse)

# Description of the Bayesian mixture model fitted in this file
# Notation:
# y_i = data point for observation i
# mu_h = Location parameter (mean) of the hth cluster
# pi_h = the weight of the hth cluster (approximately the proportion of data points in the cluster)
# sigma_h = the scale (standard deviation) parameter for the hth Gaussian distribution in the mixture

# Likelihood for two components
# y_i ~ DP = pi_1 * N(mu_1, sigma_1) + pi_2 * N(mu_2, sigma_1) + ...

# Priors - all vague
# mu_h ~ N(0, 10^-2)
# sigma_h ~ Gamma(0.1 ,0.1)
# pi_1 = V_1
# pi_h = ( V_h * (1 - V_{h-1}) * pi_{h-1} ) / V_{h-1} for h = 2, ...
# V_h ~ beta(1, a)
# Where a is the fixed concentration parameter

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T <- 100
set.seed(123)
mu_1 <- rnorm(n = 1, mean = 0, sd = 1)
mu_2 <- rnorm(n = 1, mean = 5, sd = 1)

# True values for lambda_1 = lambda_2 = 0.5
y <- c(rnorm(n = T, mean = mu_1, sd = 1), rnorm(n = T, mean = mu_2, sd = 1))

# Have a quick look at the density of y
qplot(y, geom = 'histogram')


# JAGS code ---------------------------------------------------------------

model_code <- "
model {
  for (i in 1:N) {
    y[i] ~ dnorm(mu[zeta[i]], sigma[zeta[i]]^-2)
    zeta[i] ~ dcat(pi[])
  }
  for (h in 1:H) {
    mu[h] ~ dnorm(0, 10^-2)
    sigma[h] ~ dgamma(0.1, 0.1)
  }
  # Stick breaking prior
  for (h in 1:(H-1)) {
    V[h] ~ dbeta(1,a)
  }
  V[H] <- 1
  pi[1] <- V[1]
  for (h in 2:H) {
    pi[h] <- V[h] * (1-V[h-1]) * pi[h-1] / V[h-1]
  }
}
"

# Set up the data
model_data <- list(H = 10, y = y, N = length(y), a = 1)

# Choose the parameters to watch
model_parameters <- c("pi", "mu", "sigma")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
  n.chains = 1
)

# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
# traceplot(model_run)

# Create a plot of the posterior desnsity line
post <- model_run$BUGSoutput
sigma <- post$mean$sigma
mu <- post$mean$mu
pi <- post$mean$pi

N <- length(y)
df <- data.frame(y = y)

df <- df %>%
 mutate(mixt = c(
    rnorm(n = T, mean = mu[1], sd = sqrt(sigma[1])),
    rnorm(n = T, mean = mu[2], sd = sqrt(sigma[2]))
  )) %>% as.data.frame()

df %>%
  ggplot(aes(y)) +
  geom_density(colour = "orange", size = 1) +
  geom_density(
    data = df %>% slice(1:N1),
    aes(mixt, y = ..density.. * pi[1]),
    colour = "royalblue", linetype = "dotted", size = 1.1
  ) +
  geom_density(
    data = df %>% slice(N1 + 1:N2),
    aes(mixt, y = ..density.. * pi[2]),
    colour = "plum", linetype = "dotted", size = 1.1
  ) +
  xlim(min(y) - 2, max(y) + 2) +
  annotate("text",
    x = mu[1], y = 0.21, label = expression(mu[1]),
    size = 7
  ) +
  annotate("text",
    x = mu[2], y = 0.21, label = expression(mu[2]),
    size = 7
  ) +
  theme_bw()

