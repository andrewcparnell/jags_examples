# Header ------------------------------------------------------------------

# Fitting a gaussian mixture model in JAGS
# Bruna Wundervald 

# In this file we fit a Bayesian Gaussian Mixture

# Loading in required packages
library(R2jags)
library(tidyverse)

# Description of the Bayesian Poisson model fitted in this file
# Notation:
# y_i = a mixture of gaussian distributions
# mu_i = the location of the ith distribution/cluster
# lambda_i = the weight of the ith gaussian distribution
# sigma = the dispersion parameter for each Gaussian distribution 
# in the mixture

# Likelihood
# y_i ~ GM = lambda_1 * G(mu_1, sigma) + lambda_2 * G(mu_2, sigma) 

# Priors - all vague
# mu_1 ~ G(0, 10)
# mu_2 ~ G(5, 10)
# sigma ~ InverseGamma(0.01 ,0.01)
# lambda ~ Dirichlet(1, 1)

# Posteriors - all vague
# mu_1 ~ G(0, 10)
# mu_2 ~ G(5, 10)
# sigma ~ InverseGamma(0.01 ,0.01)
# lambda ~ Dirichlet(1, 1)


# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T = 500
set.seed(123)
mu_1 = rnorm(n = 1, mean = 0, sd = 1)
mu_2 = rnorm(n = 1, mean = 5, sd = 1)

# True values for lambda_1 = lambda_2 = 0.5 
y = c(rnorm(n = T, mean = mu_1, sd = 1), rnorm(n = T, mean = mu_2, sd = 1))

Nclust = 2
N = 2*T
clust = rep(NA, N)
clust[which.min(y)] = 1 # smallest value assigned to cluster 1
clust[which.max(y)] = 2 # highest value assigned to cluster 2 
ones = rep(1, Nclust)


df <- y %>% 
  as.data.frame() 
names(df)[1] <- "y"

# Have a quick look at the density of y 
df %>% 
  ggplot(aes(y)) +
  geom_density(colour = 'orange', size = 1.3) +
  xlim(min(y)-2, max(y)+2) +
  theme_bw()


model_code = '
model {
  # Likelihood:
  for(i in 1:N) {
    y[i] ~ dnorm(mu[i] , 1/sigma_inv) 
    mu[i] <- mu_clust[clust[i]]
    clust[i] ~ dcat(lambda_clust[1:Nclust])
  }
  # Prior:
  sigma_inv ~ dgamma( 0.01 ,0.01)
  mu_clust[1] ~ dnorm(0, 10)
  mu_clust[2] ~ dnorm(5, 10)
  
  lambda_clust[1:Nclust] ~ ddirch(ones)
}
'


# Set up the data
model_data = list(Nclust = Nclust, y = y, N = N, ones = ones, clust = clust)

# Choose the parameters to watch
model_parameters =  c("sigma_inv", "mu_clust", "lambda_clust")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file = textConnection(model_code),
                 n.chains = 4,
                 n.iter = 1000,
                 n.burnin = 200,
                 n.thin = 2)


# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)
traceplot(model_run)

# Create a plot of the posterior desnsity line
post = print(model_run)
sigma = 1/post$mean$sigma_inv
mu_clust = post$mean$mu_clust
lambda_clust = post$mean$lambda_clust

N1 <- round(N * lambda_clust[1])
N2 <- round(N * lambda_clust[2])

df <- df %>%
  as.data.frame() %>% 
  mutate(mixt = c(rnorm(n = N1, mean = mu_clust[1], sd = sigma),
                  rnorm(n = N2, mean = mu_clust[2], sd = sigma)))

df %>% 
  ggplot(aes(y)) +
  geom_density(colour = 'orange', size = 1) +
  geom_density(data = df %>% slice(1:N1), 
               aes(mixt, y = ..density.. * lambda_clust[1]),
               colour = 'royalblue', linetype = 'dotted', size = 1.1) +
  geom_density(data = df %>% slice(N1+1:N2), 
               aes(mixt, y = ..density.. * lambda_clust[2]),
               colour = 'plum', linetype = 'dotted', size = 1.1) +
  xlim(min(y) - 2, max(y) + 2) +
  annotate("text", x = mu_clust[1], y = 0.21, label = expression(mu[1]),
           size = 7) +
  annotate("text", x = mu_clust[2], y = 0.21, label = expression(mu[2]),
           size = 7) + 
  theme_bw()