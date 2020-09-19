
logistic_ll <- function(y, X, par){
  #' @title logistic log likelihood
  #'
  #' @param y binary outcome 
  #' @param X design matrix 
  #' @param par vector of model coefficients 
  #' @return a scalar quantity proportional to a binomial likelihood
  #' with logistic parameterization, given y,X,and par
  sum(dbinom(y, 1, expit(X %*% par), log=TRUE))
}

normal_ll <- function(y, X, par){
  #' @title Gaussian log likelihood
  #'
  #' @param y binary outcome 
  #' @param X design matrix 
  #' @param par vector of gaussian scale parameter followed by model coefficients 
  #' @return a scalar quantity proportional to a normal likelihood
  #' with linear parameterization, given y, X, and par
  beta = par[-1]
  sig = par[1]
  if(sig<0) return(-Inf)
  sum(dnorm(y, X %*% beta, sig, log=TRUE))
}
