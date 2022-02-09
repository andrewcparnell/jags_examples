# Header ------------------------------------------------------------------

# Fitting a Gaussian mixture model in JAGS
# Bruna Wundervald & Andrew Parnell

# In this file we fit a Bayesian Gaussian Mixture model using two simple components

# Loading in required packages
library(R2jags)
library(tidyverse)

# Description of the Bayesian mixture model fitted in this file
# Notation:
# y_i = data point for observation i
# mu_k = Location parameter (mean) of the kth cluster
# lambda_k = the weight of the kth cluster (approximately the proportion of data points in the cluster)
# sigma = the scale (standaard deviation) parameter for each Gaussian distribution in the mixture

# Likelihood for two components
# y_i ~ GM = lambda_1 * N(mu_1, sigma) + lambda_2 * N(mu_2, sigma)

# Priors - all vague
# mu_k ~ N(0, 10^-2)
# sigma ~ InverseGamma(0.01 ,0.01)
# lambda_k ~ Dirichlet(1, 1, ...)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T <- 500
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
  # Likelihood:
  for(i in 1:N) {
    y[i] ~ dnorm(mu[i] , sigma^-2) 
    mu[i] <- mu_clust[clust[i]]
    clust[i] ~ dcat(lambda[1:K])
  }
  # Priors
  sigma ~ dgamma( 0.01 ,0.01)
  for (k in 1:K) {
    mu_clust_raw[k] ~ dnorm(0, 10^-2)
  }
  mu_clust <- sort(mu_clust_raw) # Ensure ordering to prevent label switching
  lambda[1:K] ~ ddirch(ones)
}
"


# Set up the data
model_data <- list(K = 2, y = y, N = length(y), ones = rep(1, K))

# Choose the parameters to watch
model_parameters <- c("sigma", "mu_clust", "lambda")

# Run the model
model_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code),
)


# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
traceplot(model_run)

# Create a plot of the posterior desnsity line
post <- model_run$BUGSoutput
sigma <- post$mean$sigma
mu_clust <- post$mean$mu_clust
lambda <- post$mean$lambda

N <- length(y)
N1 <- round(N * lambda[1])
N2 <- round(N * lambda[2])
df <- data.frame(y = y)

df <- df %>%
 mutate(mixt = c(
    rnorm(n = N1, mean = mu_clust[1], sd = sqrt(sigma)),
    rnorm(n = N2, mean = mu_clust[2], sd = sqrt(sigma))
  )) %>% as.data.frame()
  
df %>%
  ggplot(aes(y)) +
  geom_density(colour = "orange", size = 1) +
  geom_density(
    data = df %>% slice(1:N1),
    aes(mixt, y = ..density.. * lambda[1]),
    colour = "royalblue", linetype = "dotted", size = 1.1
  ) +
  geom_density(
    data = df %>% slice(N1 + 1:N2),
    aes(mixt, y = ..density.. * lambda[2]),
    colour = "plum", linetype = "dotted", size = 1.1
  ) +
  xlim(min(y) - 2, max(y) + 2) +
  annotate("text",
    x = mu_clust[1], y = 0.21, label = expression(mu[1]),
    size = 7
  ) +
  annotate("text",
    x = mu_clust[2], y = 0.21, label = expression(mu[2]),
    size = 7
  ) +
  theme_bw()


# Real example ------------------------------------------------------------
df <- iris %>%
  select(Petal.Width)

df %>%
  ggplot(aes(Petal.Width)) +
  geom_density() +
  theme_bw()


N <- dim(df)[1]
clust <- rep(NA, N)
clust[which.min(df$Petal.Width)] <- 1 # smallest value assigned to cluster 1
clust[clust == quantile(df$Petal.Width, 1 / 3)] <- 2
clust[which.max(df$Petal.Width)] <- 3 # highest value assigned to cluster 2

# Set up the data
model_data <- list(K = 3, y = df$Sepal.Length, N = N, ones = c(1, 1, 1), clust = clust)

# Choose the parameters to watch
model_parameters <- c("sigma", "mu_clust", "lambda")

# Run the model
real_data_run <- jags(
  data = model_data,
  parameters.to.save = model_parameters,
  model.file = textConnection(model_code)
)


# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(real_data_run)
print(real_data_run)
traceplot(real_data_run)

# Create a plot of the posterior desnsity line
post <- print(real_data_run)
sigma <- post$mean$sigma
mu_clust <- post$mean$mu_clust
lambda_clust <- post$mean$lambda

N1 <- round(N * lambda_clust[1])
N2 <- round(N * lambda_clust[2])
N3 <- round(N * lambda_clust[3])

df <- df %>%
  mutate(mixt1 = dnorm(df$Petal.Width, mean = mu_clust[1], sd = sqrt(sigma)),
         mixt2 = dnorm(df$Petal.Width, mean = mu_clust[2], sd = sqrt(sigma)),
         mixt3 = dnorm(df$Petal.Width, mean = mu_clust[3], sd = sqrt(sigma))
  )

df %>%
  ggplot(aes(Petal.Width)) +
  geom_density(colour = "orange", size = 1) +
  geom_line(aes(x = Petal.Width, y = mixt1*lambda_clust[1]),
               colour = "royalblue", linetype = "dotted", size = 1.1) + 
  geom_line(aes(x = Petal.Width, y = mixt2*lambda_clust[2]),
            colour = "plum", linetype = "dotted", size = 1.1) + 
  geom_line(aes(x = Petal.Width, y = mixt3*lambda_clust[3]),
            colour = "green", linetype = "dotted", size = 1.1)
