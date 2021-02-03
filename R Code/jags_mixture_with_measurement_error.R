# Header ------------------------------------------------------------------

# A 1D mixture model in JAGS with a fixed number of groups
# This version includes measurement error on the data points
# Andrew Parnell (with some help from Michael Fop at UCD)

# This model fits a mixture of normal distributions to the data given a fixed number of groups. It might be useful for clustering or other applications where standard probability distributions are not suitable
# This model assumes that the observations are observed with known noise, and that the mixtures are on the latent unobserved observations

# Some boiler plate code to clear the workspace, set the working directory, and load in required packages
rm(list = ls())
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation
# y_i = observation i, i= 1, .., N
# y_tilde_i = underlying unknown 'true' value of y
# sigma_tilde_i = known measurement error of data point i
# mu_g = mean for group g - these are sorted to avoid label switching
# sigma_g = standard deviation for group g
# z_i = group member for observation i, z_i = 1, ..., G
# pi_{ig} = probability that observation i belongs to group g. \sum_g \pi_{ig} = 1
# G = total number of groups (fixed)
# Likelihood
# y_i ~ N(y_tilde_i, sigma_tilde_i^2)
# y_tilde_i ~ N(mu[z_i], sigma[z_i]^2)
# z_i ~ categorical(pi)

# Priors
# (pi_{i1}, ..., pi_{iG}) ~ Dir(alpha) # ...is common but I prefer the below
# pi_{ig} = exp(theta_{ig}) / sum_g exp(theta_{ig}) # Use softmax rather than Dirichlet
# theta_{ig} ~ N(0, 1) # Keep reasonably close to 0. Value outside of -3/3 very extreme
# sigma_{g} ~ half-cauchy(0, 10)
# mu_g ~ N(0, 100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
G <- 3
N <- 200
set.seed(123)
sigma_tilde <- runif(N, 0.05, 0.1)
mu_g <- c(-5, 0, 2)
sigma_g <- c(1, 2, 3)
theta <- matrix(rnorm(N * G, 0, 3), ncol = G, nrow = N)
pi <- exp(theta) / apply(exp(theta), 1, sum)
Z <- rep(NA, N)
for (i in 1:N) Z[i] <- sample(1:G, size = 1, prob = pi[i, ])
y_tilde <- rnorm(N, mu_g[Z], sigma_g[Z])
y <- rnorm(N, y_tilde, sigma_tilde)

# Create a quick plot
hist(y, breaks = 30, freq = FALSE)
for (g in 1:G) curve(dnorm(x, mean = mu_g[g]) / G, col = g, add = TRUE)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code <- "
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dnorm(mu_g[Z[i]], tau[i])
    tau[i] = (sigma_g[Z[i]]^2 + sigma_tilde[i]^2)^-1
    log_lik[i] <- logdensity.norm(y[i], mu_g[Z[i]], tau[i])
    Z[i] ~ dcat(pi[i, 1:G])

    for (g in 1:G) {
      exp_theta[i, g] <- exp(theta[i, g])
      pi[i, g] <- exp(theta[i, g]) / sum(exp_theta[i, 1:G])
      theta[i, g] ~ dnorm(0, 6^-2)
    }
  }

  # Priors
  for (g in 1:G) {
    mu_g_raw[g] ~ dnorm(0, 100^-2)
    sigma_g[g] ~ dt(0, 10^-2, 1)T(0,)
  }
  # Make sure these are in order to avoid label switching
  mu_g <- sort(mu_g_raw[1:G])

}
"

# Set up the data
model_data <- list(N = N, y = y, sigma_tilde = sigma_tilde, G = G)

# Choose the parameters to watch
model_parameters <- c("mu_g", "sigma_g", "Z", "pi")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code)
)

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
plot(model_run)
print(model_run)

# Compare model output with truth
model_run$BUGSoutput$mean$mu_g
# [1] -5.0237053 -0.2241286  2.3506181
model_run$BUGSoutput$mean$sigma_g
# [1] 0.9150342 2.4221499 3.2985686
mu_g
sigma_g

# Real example ------------------------------------------------------------
