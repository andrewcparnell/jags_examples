
# --------------------------------------------------------------- #
# Description: Metropolis-Hastings for Bayesian linear regression #
#              with a Laplace as prior for betas and an Uniform   #
#              for sigma2                                         #
# Author: Alan Inglis                                             #
# Last modification: 18/12/2018                                   #
# --------------------------------------------------------------- #

library(MASS)
library(ggplot2)
library(gridExtra)
library(progress)

## Defining simulation size, burn-in, etc
## --------------------------------------
Niter <- 10000
BurnIn <- 5000
TotIter <- Niter+BurnIn
N <- 10000
alpha <- -2
beta <- 3
P <- c(alpha,beta)
sigma <- 3
js = 0
jb = 0
AuxBurnIn <- 1

## Simulate data
## -----------------
set.seed(123)
x0 <- rep(1,N)
x1 <- rnorm(n = N, 0, 1)
X <- cbind(x0,x1)
y <- rnorm(n = N, mean = P[1] + P[2]*x1, sd = sigma)

## Defining prior distributions and values of hyperparameters
## ----------------------------------------------------------
mi = rbind(0,0)
vi = 100  

## Sigma2 ~ U(a,b)
# -----------------------------
a <- 0
b <- 10


## Data frame that will store MCMC values for betas
## ------------------------------------------------
SaveResults <- as.data.frame(matrix(data = NA, nrow = Niter, ncol = length(P)+2))
colnames(SaveResults) <- c('Iter', 'Alpha', 'Beta', 'Sigma')


## Initial values 
## ---------------------------------------------------------------------------
MCMCBetasI <- c(10,10)
V = diag(2)*0.0005 # Constructs a diagonal matrix

sigma2_ = 1

# Since the proposal distribution for sigma2 is a Gamma, the idea is as following:
# mean = alpha/beta
# var = alpha/beta^2; Thus
# alpha = mean^2 / var;
# beta = mean/var.    
# Consider that sigma2_ = mean and var = var, where var is set up below.

var = 0.5 # Controlling the variance of the proposal distribution for sigma2. Thus, it is possible to control the acceptance rate of sigma2.

## MC:
## -----------------------------------
pb <- progress_bar$new(
  format = "  iterations [:bar] :percent done",
  total = TotIter, clear = FALSE, width= 60) # Displays a progress bar for iterations

for(i in 1:TotIter){
  pb$tick()
  
  
  # Proposal distribution for sigma2 
  sigma2c = rgamma(1, (sigma2_^2)/var,  scale=1/(sigma2_ / var))
  
  
  # Computing the ratio in the Metropolis-Hastings
  ratio_sigma2 = (((-N/2) * log(sigma2c) - 0.5/sigma2c * (t(y - (X%*%MCMCBetasI))%*%(y-(X%*%MCMCBetasI))) - log(b - a) + 1/sigma2_ + dgamma(sigma2_, (sigma2c^2)/var, scale = 1/(sigma2c / var))) -
                  ((-N/2) * log(sigma2_) - 0.5/sigma2_ * (t(y - (X%*%MCMCBetasI))%*%(y-(X%*%MCMCBetasI))) - log(b - a) + 1/sigma2c + dgamma(sigma2c, (sigma2_^2)/var, scale = 1/(sigma2_ / var))))
  
  # Accept/Reject step for sigma2
  if(runif(1) < min(1, exp(ratio_sigma2)))
  {
    sigma2_ = sigma2c
    js = js +1
  }
  
  # Proposal distribution for beta
  MCMCBetasC <- mvrnorm(1, MCMCBetasI, V)
  
  # Computing the ratio:
  ratio_beta <-  ((-0.5/sigma2_* (t(y - (X %*% MCMCBetasC)) %*% (y - (X %*% MCMCBetasC))  - 0.5/vi*sum(abs(MCMCBetasC - mi))^2)) -
                  (-0.5/sigma2_* (t(y - (X %*% MCMCBetasI)) %*% (y - (X %*% MCMCBetasI))  - 0.5/vi*sum(abs(MCMCBetasI - mi))^2)))
  
  # Accept/Reject step for beta
  if(runif(1) < min(1, exp(ratio_beta)))
  {
    MCMCBetasI <- MCMCBetasC
    jb <- jb+1
  }
  
  if (i > BurnIn){
    SaveResults[AuxBurnIn,] <- c(AuxBurnIn, MCMCBetasI, sigma2_)
    AuxBurnIn <- AuxBurnIn + 1
  }
}

## Acceptance rate
acceptance_rate_sigma <- js/TotIter
acceptance_rate_beta  <- jb/TotIter

acceptance_rate_beta
acceptance_rate_sigma

## Plots
## -----
Chainalpha <- ggplot(SaveResults, aes(x=Iter)) +
  geom_line(aes(y=Alpha)) +
  labs(title=expression(paste('Chain of ', alpha)),
       x='Iterations',
       y='Values') +
  geom_hline(yintercept= P[1], linetype = 'dotted', color='coral') +
  theme_bw()

Chainbeta <- ggplot(SaveResults, aes(x=Iter)) +
  geom_line(aes(y=Beta)) +
  labs(title=expression(paste('Chain of ', beta)),
       x='Iterations',
       y='Values') +
  geom_hline(yintercept= P[2], linetype = 'dotted', color='coral') +
  theme_bw()


ChainS2 <- ggplot(SaveResults, aes(x=Iter)) +
  geom_line(aes(y=Sigma)) +
  labs(title=expression(paste('Chain of ', sigma^2)),
       x='Iterations',
       y='Values') +
  geom_hline(yintercept=(sigma^2), linetype = 'dotted', color='coral') +
  theme_bw()

grid.arrange(Chainalpha, Chainbeta, ChainS2, ncol = 2)

# Comparison: 
#---------------------------------------------------------------------
# Checking the posterior estimates ------------------
alpha_est <- mean(SaveResults$Alpha)
beta_est <- mean(SaveResults$Beta)
sigma_est <- mean(SaveResults$Sigma)

means_MCMC = c(alpha_est, beta_est, sigma_est)
means_JAGS = c(-1.98, 2.90, 8.2369)
sd_JAGS = c(0.09098112, 0.09166854, 0.06481725)

# load needed libraries-----------------------------
library(reshape2)

# Data Frame----------------------------------------
Name1 <- c('Alpha', 'Beta', 'Sigma^2')
myData <- data.frame( Name1, means_JAGS, means_MCMC)
myData

# reshape data into long format--------------------
myDATAlong <- melt(myData)

# make the plot------------------------------------
ggplot(myDATAlong) +
  geom_bar(aes(x = Name1, y = value, fill = variable), 
           stat="identity", position = "dodge", width = 0.7) +
  scale_fill_manual("Linear Regression \nMeans\n", values = c("red","blue"), 
                    labels = c("Jags", "MCMC")) +
  labs(x="\nVars",y="Mean\n") +
  theme_bw(base_size = 14)
