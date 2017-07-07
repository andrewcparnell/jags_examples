# JAGS examples

A large set of JAGS examples using R. They include linear regression, generalised linear modelling, hierarchical models, non-parametric smoothing (Gaussian Processes and splines), time series models (discrete and continuous, univariate and multivariate), change-point analysis, and others.

I find that JAGS is a fantastic tool for fast prototyping or basic analysis of data sets. However, I find that many of the examples I find online just jump straight into the code without any explanation of the mathematical details, nor examples of how it’s used. This set of scripts is designed to help those who want to quickly grab something off the shelf and use or amend it. 

Each script is set out with the following structure:

- Boiler plate code to clear workspace and load packages
- Mathematical description of the model
- Code to simulate some data
- JAGS code for the model
- R code to fit the model
- Output analysis and plots
- A real data example

To get started you will need to have both [R](http://www.r-project.org) and [JAGS](http://sourceforge.net/projects/mcmc-jags/files/) installed. 

If you would like to contribute to the repository please keep your files in the same structure as above. A file `jags_template.R` can be used to fill in your new code.
