

print.metropolis.samples <- function(x, ...){
  #' @title Print a metropolis.samples object
  #'
  #' @description This function allows you to summarize output from the "metropolis_glm" function.
  #' @details None 
  #' @param x a "metropolis.samples" object from the function "metropolis_glm"
  #' @param ... not used.
  #' @export
  #' @return An unmodified "metropolis.samples" object (invisibly)
  fam = x$family$family
  mod = as.character(x$f)
  fun = paste(mod[2], mod[1], mod[3])
  cat(paste0("\n Metropolis: ", x$iter, " iterations with ", x$burnin, " burn-in", "\n"))
  cat(paste0(" Model: ", fun, "    (", fam, ")\n"))
  cat(paste0(" Guided: ", x$guided,  "\n"))
  cat(paste0(" Adaptive: ", x$adaptive,  "\n"))
  invisible(x)    
}


summary.metropolis.samples <- function(object, keepburn=FALSE, ...){
  #' @title Summarize a probability distribution from a Markov Chain
  #'
  #' @description This function allows you to summarize output from the metropolis function.
  #' @details TBA
  #' @param object an object from the function "metropolis"
  #' @param keepburn keep the burnin iterations in calculations (if adapt=TRUE, keepburn=TRUE 
  #' will yield potentially invalid summaries)
  #' @param ... not used
  #' @return returns a list with the following fields:
  #'     nsamples: number of simulated samples 
  #'     sd: standard deviation of parameter distributions 
  #'     se: standard deviation of parameter distribution means 
  #'     ESS_parms: effective sample size of parameter distribution means 
  #'     postmean: posterior means and normal based 95% credible intervals 
  #'     postmedian: posterior medians and percentile based 95% credible intervals 
  #'     postmode: posterior modes and highest posterior density based 95% credible intervals
  #' @export
  #' @examples
  #' dat = data.frame(y = rbinom(100, 1, 0.5), x1=runif(100), x2 = runif(100))
  #' res = metropolis_glm(y ~ x1 + x2, data=dat, family=binomial(), iter=10000, burnin=3000, 
  #' adapt=TRUE, guided=TRUE, block=FALSE)
  #' summary(res)
  
  # function to compute summary of probability distribution sampled via
  # metropolis-hastings algorithm
  samples = object$parms[object$parms$burn<ifelse(keepburn, 2, 1),]
  sims = samples[, -grep("burn|iter", names(samples))]
  nms = object$dimnames
  if(object$family$family=="gaussian") nms = c("logsigma", nms)
  names(sims) <- nms
  nsims = dim(sims)[1]
  
  #effective sample size 
  # geyer 1992 (not yet finished)
  #autocorr = apply(sims, 2, function(x) ar(x, aic=FALSE, order.max=50)$ar)
  #autocorrs = apply(autocorr, 2, function(x) cumsum(c(0, diff(c(0,diff(x<0))))==-1))
  # coda package
  ar.parms = apply(sims, 2, ar)
  sv.parms = sapply(ar.parms, function(x) x$var.pred/(1-sum(x$ar))^2)
  var.parms = apply(sims, 2, var)
  ess.parms = pmin(nsims, nsims*var.parms/sv.parms)

  #estimates based on posterior mean, normal approx probability intervals
  (mean.parms  <- apply(sims, 2, mean))
  (sd.parms  <- sqrt(var.parms))
  lci.normal.parms <- mean.parms-1.96*sd.parms
  uci.normal.parms <- mean.parms+1.96*sd.parms

  #estimates based on posterior median, quantile based probability intervals
  median.parms = apply(sims, 2, function(x) quantile(x, c(0.5)))
  lci.quantile.parms <- apply(sims, 2, function(x) quantile(x, c(0.025)))
  uci.quantile.parms <- apply(sims, 2, function(x) quantile(x, c(0.975)))

  #estimates based on posterior mode (via kernal density), highest probability density based intervals
  dens.parms <- apply(sims, 2, density) 
  #mode.parms = sapply(dens.parms, function(x) x$x[which.max(x$y)][1])
  mode.parms = as.numeric(sims[which.max(object$lpost[object$parms$burn<ifelse(keepburn, 2, 1)]),][1,])
  names(mode.parms) <- names(mean.parms)
  hpdfun <- function(x, p=0.95){
   x = sort(x)
   xl = length(x)
   g = max(1, min(xl-1, round(xl*p)))
   init <- 1:(xl - g)
   idx = which.min(x[init + g] - x[init])
   c(x[idx], x[idx+g])
  }
  hpd.parms = apply(sims, 2, function(x) hpdfun(x, c(0.95)))
  lci.hpd.parms <- hpd.parms[1,]
  uci.hpd.parms <- hpd.parms[2,]

  list(
    nsamples = nsims,
    sd = sd.parms,
    se = sd.parms/sqrt(ess.parms),
    ESS_parms = ess.parms,
     postmean = cbind(mean=mean.parms, normal_lci=lci.normal.parms, normal_uci=uci.normal.parms),
     postmedian = cbind(median = median.parms, pctl_lci = lci.quantile.parms, pctl_uci = uci.quantile.parms),
     postmode = cbind(mode=mode.parms, hpd_lci = lci.hpd.parms, hpd_uci = uci.hpd.parms)
     )
}

plot.metropolis.samples <- function(x, keepburn=FALSE, parms=NULL, ...){
  #' @title Plot the output from the metropolis function
  #'
  #' @description This function allows you to summarize output from the metropolis function.
  #' @details TBA
  #' @param x the outputted object from the "metropolis_glm" function
  #' @param keepburn keep the burnin iterations in calculations (if adapt=TRUE, keepburn=TRUE 
  #' @param parms names of parameters to plot (plots the first by default, if TRUE, plots all)
  #' @param ... other arguments to plot
  #' @import stats graphics
  #' @importFrom grDevices dev.hold
  #' @return None
  #' @export
  #' @examples
  #' dat = data.frame(y = rbinom(100, 1, 0.5), x1=runif(100), x2 = runif(100))
  #' res = metropolis_glm(y ~ x1 + x2, data=dat, family=binomial(), iter=10000, burnin=3000, 
  #' adapt=TRUE, guided=TRUE, block=FALSE)
  #' plot(res)
  opar <- par(no.readonly = TRUE)       
  on.exit(par(opar))                  
  samples = x$parms[x$parms$burn<ifelse(keepburn, 2, 1),]
  sims = samples[, -grep("burn|iter", names(samples))]
  nms = x$dimnames
  if(x$family$family=="gaussian") nms = c("logsigma", nms)
  names(sims) <- nms
  #nsims = dim(sims)[1]
  if(isTRUE(parms[1])) parms = names(sims)
  if(is.null(parms[1])) parms = 1
  
  plsims = sims[,parms, drop=FALSE]
  if(!keepburn) idx=(x$burn+1):(x$burn+x$iter)
  if(keepburn) idx=1:(x$burn+x$iter)
  par(mfcol=c(2,1), mar=c(3,4,1,1))
  for(k in 1:dim(plsims)[2]){
    #scatter plot
    hist(plsims[,k], main="", breaks = 50)
    #dev.hold()
    #trace plot
    plot(idx, plsims[,k], type="l", ylab=names(plsims)[k], ...)
  }
}


metropolis.control <- function(
  adapt.start = 25,
  adapt.window = 200, 
  adapt.update = 25,
  min.sigma = 0.001,
  prop.sigma.start = 1,
  scale = 2.4 
  ){
  #' @title metropolis.control 
  #'
  #' @param adapt.start start adapting after this many iterations; set to iter+1 to turn off adaptation 
  #' @param adapt.window base acceptance rate on maximum of this many iterations
  #' @param adapt.update frequency of adaptation
  #' @param min.sigma minimum of the proposal distribution standard deviation (if set to zero, 
  #' posterior may get stuck)
  #' @param prop.sigma.start starting value, or fixed value for proposal distribution s
  #' standard deviation
  #' @param scale scale value for adaptation (how much should the posterior
  #'  variance estimate be scaled by?). Scale/sqrt(p) is used in metropolis_glm function, and 
  #'  Gelman et al. (2014, ISBN: 9781584883883) recommend a scale of 2.4
  #'  @return A list of parameters used in fitting with the following named objects
  #'    adapt.start, adapt.window,adapt.update,min.sigma,prop.sigma.start,scale
  #' @export
  list(
    adapt.start = adapt.start, 
    adapt.window = adapt.window,
    adapt.update = adapt.update,
    min.sigma = min.sigma,
    prop.sigma.start = prop.sigma.start,
    scale = scale 
  )
}


calcpost <- function(y,X,par,family, pm, pv){
  if(family$family == "binomial") llp = logistic_ll(y, X, par)
  if(family$family == "gaussian"){
    modpar = par
    modpar[1] = exp(par[1])
    llp = normal_ll(y, X, modpar)
  }
  # add in priors
  if(!is.null(pm)){
    llp = llp + sum(dnorm(par, mean = pm ,sd = sqrt(pv), log=TRUE))
  }
  # add in Jacobian of transform for linear models due to transformed scale parameter
  # posterior*J -> llp + log(J)
  #if(family$family == "gaussian") llp = llp + exp(par[1])
  if(family$family == "gaussian") llp = llp + par[1]
  llp
}


.metropolis_glm <- function(
  f, 
  data,
  family=binomial(),
  iter=100, 
  burnin=round(iter/2), 
  pm=NULL, 
  pv=NULL, 
  chain=1, 
  prop.sigma.start=.1, 
  inits=NULL, 
  adaptive=TRUE, 
  guided=FALSE, 
  block=TRUE,
  saveproposal=FALSE,
  control = metropolis.control()
) {
  
  # error catching 
  if (is.character(family)) 
    family <- get(family, mode = "function", envir = parent.frame())
  if (is.function(family)) 
    family <- family()
  if (is.null(family$family)) {
    stop("'family' not recognized")
  }
  if (!(family$family %in% c("gaussian", "binomial"))) {
    stop("only 'gaussian' and 'binomial' families are currently implemented")
  }
  if(length(block)>1 & guided){
    warning("Block sampling with 'guided=TRUE' is not advised, unless block=TRUE")
  }
  # collect some info on data
  X = model.matrix(f, data = data)
  mterm = terms(f)
  outcol = as.character(attr(mterm, "variables")[[2]])
  y = data[,outcol]
  
  ###########3
  p = dim(X)[2] # number of parameters
  if(family$family == "gaussian"){
    p = p+1 # scale parameter
  }
  # create empty matrixes/vectors for posterior estimates
  accept <- parms <- matrix(nrow=iter+burnin, ncol=p)
  lpost <- numeric(iter+burnin)
  proposals = NULL
  # generate initial values
  if(is.null(inits)){
    inits = c(beta=runif(p)*4-2)
  } 
  if(length(inits)==1 && inits=="glm"){
    mlefit = glm(f, family=family, data)
    inits = as.numeric(mlefit$coefficients)
    if(family$family == "gaussian") inits = c(logscale=log(sd(mlefit$residuals)), beta=inits)
  }
  parms[1,] <- inits
  lpost[1] <- calcpost(y,X,inits,family, pm, pv)  
  if(saveproposal & length(block)==1 && block) proposals = parms
  if(length(control$prop.sigma.start)==1){
    cov = rep(control$prop.sigma.start, p)
  }else{
    cov = control$prop.sigma.start
  }
  
  accept[1,] = rep(1, p)
  if(!adaptive) control$adapt.start = iter+1+burnin
  # guiding direction
  if(length(block)==1 && block) dir = 1
  if(length(block)>1) dir = rep(1, length(block))
  if(length(block)==1 && !block) dir = rep(1, p)
  # run metropolis algorithmm
  for(i in 2:(iter+burnin)){
    #if adaptive, then adapt while in the adaptation phase
    if((i>=control$adapt.start  & !(i%%control$adapt.update) & i<=burnin)){
      cov <- control$scale/sqrt(p)*apply(parms, 2, sd, na.rm=TRUE)
    }
    b.prv = parms[i-1,]
    b.cand = b.prv
    if(length(block)==1 && block){
      # block update all parameters
      # non-normalized log-probability at previous beta
      llp = calcpost(y,X,b.prv,family, pm, pv)        
      # draw candidate value from proposal distribution
      z = rnorm(p,0, cov) 
      if(guided) z = abs(z) * dir
      b.cand = b.cand + z
      if(saveproposal) proposals[i,] = b.cand
      
      # non-normalized log-probability at candidate beta
      lp = calcpost(y,X,b.cand,family, pm, pv)        
      # accept/reject candidate value
      accept.prob = exp(lp-llp) # acceptance probability
      a = (accept.prob > runif(1))
      if(is.nan(a)) stop("Acceptance probability is NaN, indicating a problem with the chain. 
                           If using adapt=TRUE, try changing some of the adaptation parameters 
                           in 'control' (see help file for metropolis.control and metropolis_glm)")
      if(!a){
        b.cand = b.cand - z
        dir = dir
      }
      #lpost[i] <- ifelse(a, lp, llp)
      accept[i,] = a
    } else{ # partial blocks, or update 1 by 1
      # loop over parameters
      if(length(block)>1) blocks=block else blocks = rep(1,p)
      st = cumsum(c(1, blocks))
      end = cumsum(c(blocks, 1))
      blocklist = list()
      for(idx in 1:length(st[-1])){
        blocklist[[idx]] = st[idx]:end[idx]
      }
      for(q in blocklist){ # loop over blocks
        # non-normalized log-probability at previous beta
        llp = calcpost(y,X,b.prv,family, pm, pv)        
        # draw candidate value from proposal distribution
        z = rep(0, p)
        z[q] = rnorm(length(q),0, cov[q]) 
        if(guided) z = abs(z) * dir
        b.cand = b.cand + z
        # non-normalized log-probability at candidate beta
        lp = calcpost(y,X,b.cand,family, pm, pv)        
        # accept/reject candidate value
        accept.prob = exp(lp-llp) # acceptance probability
        a = (accept.prob > runif(1))
        if(is.nan(a)) stop("Acceptance probability is NaN, indicating a problem with the chain. 
                           If using adapt=TRUE, try changing some of the adaptation parameters 
                           in 'control' (see help file for metropolis.control and metropolis_glm)")
        if(!a){
          b.cand = b.cand - z
          dir[q] = -dir[q]
        }
        b.prv[q] = b.cand[q]
        accept[i,q] = a
      }# end loop over blocks
    } # end partial blocks, or update 1 by 1
    lpost[i] <- ifelse(a, lp, llp)
    parms[i,] = b.cand
  } #loop over i
  bt = data.frame(parms)
  if(family$family=="binomial") nm = paste0('b_', 0:(p-1), '')
  if(family$family=="gaussian") nm = c("logsigma", paste0('b_', 0:(p-2), ''))
  names(bt) <- nm
  bt$iter = 1:dim(bt)[1]
  bt$burn = 1.0*(bt$iter<=burnin)
  bt <- bt[,c('iter', 'burn', grep('logsigma|b_', names(bt), value = TRUE, fixed=FALSE))]
  # output an R list with posterior samples, acceptance rate, and covariance of proposal distribution
  res = list(parms = bt, 
             accept=accept,
             lpost=lpost,
             cov=cov, 
             iter=iter, 
             burnin=burnin, 
             f=f, 
             family=family, 
             dimnames = colnames(X), 
             guided=guided,
             adaptive = adaptive,
             priors = ifelse(is.null(pm), "uniform", "normal"),
             pm = pm,
             pv = pv,
             inits = inits,
             proposals = proposals,
             adaptive = adaptive,
             guided=guided,
             block=block
  )
  class(res) <- "metropolis.samples"
  res
}

#' @title Use the Metropolis Hastings algorithm to estimate Bayesian glm parameters
#'
#' @description This function carries out the Metropolis algorithm.
#' @details Implements the Metropolis algorithm, which allows user specified proposal distributions
#'  or implements an adaptive algorithm as described by 
#'  Gelman et al. (2014, ISBN: 9781584883883).
#'  This function also allows the "Guided" Metropolis algorithm of 
#'  Gustafson (1998) \doi{doi:10.1023/A:1008880707168}. Note that by default all
#'  parameters are estimated simulataneously via "block" sampling, but this
#'  default behavior can be changed with the "block" parameter. When using 
#'  guided=TRUE, block should be set to FALSE.
#' @param f an R style formula (e.g. y ~ x1 + x2)
#' @param data an R data frame containing the variables in f
#' @param family R glm style family that determines model form: gaussian() or binomial()
#' @param iter number of iterations after burnin to keep
#' @param burnin number of iterations at the beginning to throw out (also used for adaptive phase)
#' @param pm vector of prior means for normal prior on log(scale) (if applicable) and 
#' regression coefficients (set to NULL to use uniform priors) 
#' @param pv vector of prior variances for normal prior on log(scale) (if applicable) and 
#' regression coefficients (set to NULL to use uniform priors) 
#' @param chain chain id (plan to deprecate)
#' @param prop.sigma.start proposal distribution standard deviation (starting point if adapt=TRUE) 
#' @param inits NULL, a vector with length equal to number of parameters (intercept + x + scale 
#' ;gaussian() family only model only), or "glm" to set priors based on an MLE fit
#' @param adaptive logical, should proposal distribution be adaptive? (TRUE usually gives better answers)
#' @param guided logical, should the "guided" algorithm be used (TRUE usually gives better answers)
#' @param block logical or a vector that sums to total number of parameters (e.g. if there are 4 
#' random variables in the model, including intercept, then block=c(1,3) will update the 
#' intercept separately from the other three parameters.) If TRUE, then updates each parameter 
#' 1 by 1. Using `guide`=TRUE with `block` as a vector is not advised
#' @param saveproposal (logical, default=FALSE) save the rejected proposals (block=TRUE only)?
#' @param control parameters that control fitting algorithm. See metropolis.control()
#' @return An object of type "metropolis.samples" which is a named list containing posterior
#' MCMC samples as well as some fitting information.
#' @import stats
#' @export
#' @examples
#' dat = data.frame(y = rbinom(100, 1, 0.5), x1=runif(100), x2 = runif(100))
#' \donttest{
#' res = metropolis_glm(y ~ x1 + x2, data=dat, family=binomial(), iter=1000, burnin=3000, 
#' adapt=TRUE, guided=TRUE, block=FALSE)
#' res
#' summary(res)
#' apply(res$parms, 2, mean)}
#' glm(y ~ x1 + x2, family=binomial(), data=dat)
#' dat = data.frame(y = rnorm(100, 1, 0.5), x1=runif(100), x2 = runif(100), x3 = rpois(100, .2))
#' \donttest{
#' res = metropolis_glm(y ~ x1 + x2 + factor(x3), data=dat, family=gaussian(), inits="glm", 
#' iter=10000, burnin=3000, adapt=TRUE, guide=TRUE, block=FALSE)
#' apply(res$parms, 2, mean)
#' glm(y ~ x1 + x2+ factor(x3), family=gaussian(), data=dat)}
metropolis_glm <- function(
                       f, 
                       data,
                       family=binomial(),
                       iter=100, 
                       burnin=round(iter/2), 
                       pm=NULL, 
                       pv=NULL, 
                       chain=1, 
                       prop.sigma.start=.1, 
                       inits=NULL, 
                       adaptive=TRUE, 
                       guided=FALSE, 
                       block=TRUE,
                       saveproposal=FALSE,
                       control = metropolis.control()
                       ) {

  # error catching 
  if (is.character(family)) 
    family <- get(family, mode = "function", envir = parent.frame())
  if (is.function(family)) 
    family <- family()
  if (is.null(family$family)) {
    stop("'family' not recognized")
  }
  if (!(family$family %in% c("gaussian", "binomial"))) {
    stop("only 'gaussian' and 'binomial' families are currently implemented")
  }
  if(length(block)>1 & guided){
    warning("Block sampling with 'guided=TRUE' is not advised, unless block=TRUE")
  }
  # collect some info on data
    X = model.matrix(f, data = data)
    mterm = terms(f)
    outcol = as.character(attr(mterm, "variables")[[2]])
    y = data[,outcol]
    p = dim(X)[2] # number of parameters
    if(family$family == "gaussian"){
      p = p+1 # scale parameter
    }
    # create empty matrixes/vectors for posterior estimates
    accept <- parms <- matrix(nrow=iter+burnin, ncol=p)
    lpost <- numeric(iter+burnin)
    proposals = NULL
  # generate initial values
    if(is.null(inits)){
      inits = c(beta=runif(p)*4-2)
    } 
    if(length(inits)==1 && inits=="glm"){
      mlefit = glm(f, family=family, data)
      inits = as.numeric(mlefit$coefficients)
      if(family$family == "gaussian") inits = c(logscale=log(sd(mlefit$residuals)), beta=inits)
    }
    parms[1,] <- inits
    lpost[1] <- calcpost(y,X,inits,family, pm, pv)  
    if(saveproposal & length(block)==1 && block) proposals = parms
    if(length(control$prop.sigma.start)==1){
      cov = rep(control$prop.sigma.start, p)
    }else{
      cov = control$prop.sigma.start
    }

    accept[1,] = rep(1, p)
    if(!adaptive) control$adapt.start = iter+1+burnin
    # guiding direction
    if(length(block)==1 && block) dir = 1
    if(length(block)>1) dir = rep(1, length(block))
    if(length(block)==1 && !block) dir = rep(1, p)
    # run metropolis algorithmm
  for(i in 2:(iter+burnin)){
   #if adaptive, then adapt while in the adaptation phase
    if((i>=control$adapt.start  & !(i%%control$adapt.update) & i<=burnin)){
      cov <- control$scale/sqrt(p)*apply(parms, 2, sd, na.rm=TRUE)
    }
    b.prv = parms[i-1,]
    b.cand = b.prv
    if(length(block)==1 && block){
      # block update all parameters
      # non-normalized log-probability at previous beta
      llp = calcpost(y,X,b.prv,family, pm, pv)        
      # draw candidate value from proposal distribution
      z = rnorm(p,0, cov) 
      if(guided) z = abs(z) * dir
      b.cand = b.cand + z
      if(saveproposal) proposals[i,] = b.cand
      
      # non-normalized log-probability at candidate beta
      lp = calcpost(y,X,b.cand,family, pm, pv)        
      # accept/reject candidate value
      accept.prob = exp(lp-llp) # acceptance probability
      a = (accept.prob > runif(1))
      if(is.nan(a)) stop("Acceptance probability is NaN, indicating a problem with the chain. 
                           If using adapt=TRUE, try changing some of the adaptation parameters 
                           in 'control' (see help file for metropolis.control and metropolis_glm)")
      if(!a){
        b.cand = b.cand - z
        dir = dir
      }
      #lpost[i] <- ifelse(a, lp, llp)
      accept[i,] = a
    } else{ # partial blocks, or update 1 by 1
      # loop over parameters
      if(length(block)>1) blocks=block else blocks = rep(1,p)
      st = cumsum(c(1, blocks))
      end = cumsum(c(blocks, 1))
      blocklist = list()
      for(idx in 1:length(st[-1])){
        blocklist[[idx]] = st[idx]:end[idx]
      }
      for(q in blocklist){ # loop over blocks
        # non-normalized log-probability at previous beta
        llp = calcpost(y,X,b.prv,family, pm, pv)        
        # draw candidate value from proposal distribution
        z = rep(0, p)
        z[q] = rnorm(length(q),0, cov[q]) 
        if(guided) z = abs(z) * dir
        b.cand = b.cand + z
        # non-normalized log-probability at candidate beta
        lp = calcpost(y,X,b.cand,family, pm, pv)        
        # accept/reject candidate value
        accept.prob = exp(lp-llp) # acceptance probability
        a = (accept.prob > runif(1))
        if(is.nan(a)) stop("Acceptance probability is NaN, indicating a problem with the chain. 
                           If using adapt=TRUE, try changing some of the adaptation parameters 
                           in 'control' (see help file for metropolis.control and metropolis_glm)")
        if(!a){
          b.cand = b.cand - z
          dir[q] = -dir[q]
        }
        b.prv[q] = b.cand[q]
        accept[i,q] = a
      }# end loop over blocks
    } # end partial blocks, or update 1 by 1
    lpost[i] <- ifelse(a, lp, llp)
    parms[i,] = b.cand
  } #loop over i
  bt = data.frame(parms)
  if(family$family=="binomial") nm = paste0('b_', 0:(p-1), '')
  if(family$family=="gaussian") nm = c("logsigma", paste0('b_', 0:(p-2), ''))
  names(bt) <- nm
  bt$iter = 1:dim(bt)[1]
  bt$burn = 1.0*(bt$iter<=burnin)
  bt <- bt[,c('iter', 'burn', grep('logsigma|b_', names(bt), value = TRUE, fixed=FALSE))]
  # output an R list with posterior samples, acceptance rate, and covariance of proposal distribution
  res = list(parms = bt, 
             accept=accept,
             lpost=lpost,
             cov=cov, 
             iter=iter, 
             burnin=burnin, 
             f=f, 
             family=family, 
             dimnames = colnames(X), 
             guided=guided,
             adaptive = adaptive,
             priors = ifelse(is.null(pm), "uniform", "normal"),
             pm = pm,
             pv = pv,
             inits = inits,
             proposals = proposals,
             adaptive = adaptive,
             guided=guided,
             block=block
             )
  class(res) <- "metropolis.samples"
  res
}



# sample.metropolis.samples <- function(x, iter, data){
#   existsamp = x$parms[,grep('logsigma|b_', names(x$parms), value = TRUE, fixed=FALSE)]
#   newsamps = metropolis_glm(
#     f=x$f, 
#     data,
#     family=x$family,
#     iter=iter, 
#     burnin=0, 
#     pm=x$pm, 
#     pv=x$pv, 
#     chain=1, 
#     prop.sigma.start=.1, 
#     inits=existsamp[dim(existsamp)[1],], 
#     adaptive=TRUE, 
#     guided=FALSE, 
#     block=TRUE,
#     saveproposal=FALSE,
#     control = metropolis.control(prop.sigma.start = x$cov),
#     adaptive = x$adaptive,
#     guided = x$guided,
#     block = x$block
#   )
# }


