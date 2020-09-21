#### convert to other types ####

# coda object
as.mcmc.metropolis.samples <- function(x, ...){
  #' @title Convert glm_metropolis output to `mcmc` object from package coda
  #'
  #' @description Allows use of useful functions from `coda` package
  #' @details TBA
  #' @param x an object from the function "metropolis"
  #' @param ... not used
  #' @return An object of type "mcmc" from the coda package
  #' @importFrom coda mcmc as.mcmc
  #' @export
  #' @method as.mcmc metropolis.samples
  #' @examples
  #' \dontrun{
  #' library("coda")
  #' dat = data.frame(y = rbinom(100, 1, 0.5), x1=runif(100), x2 = runif(100))
  #' res = metropolis_glm(y ~ x1 + x2, data=dat, family=binomial(), iter=10000, burnin=3000, 
  #' adapt=TRUE, guided=TRUE, block=FALSE)
  #' res2 = as.mcmc(res)
  #' summary(res2)
  #' }
  samples = x$parms[x$parms$burn==0,grep('logsigma|b_', names(x$parms), value = TRUE, fixed=FALSE)]
  mcmc(data= samples, 
       start =  x$burnin, 
       end = x$iter,
       thin = 1)
  
}
