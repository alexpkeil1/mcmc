library(ggplot2)
library(rjags)
library(rstan)
#leading to 3 exposed cases and 5 exposed controls
# 36 leukemia cases and 198 controls

y = c(rep(1, 36), rep(0, 198))
x = c(rep(1, 3), rep(0, 33), rep(1, 5), rep(0, 193))

table(x,y)



expit <- function(mu) 1/(1+exp(-mu))
mh.adaptive.logistic <- function(iter, y, X=Xs, chain=1, prop.sigma.start=1, inits=NULL) {
  #adaptive metropolis hastings
  p = dim(X)[2]
  accept <- beta <- matrix(nrow=iter, ncol=p)
  if(is.null(inits)) inits = runif(p)*4-2
  beta[1,] = inits
  cov =  rep(prop.sigma.start, p)# starting value for covariance parameter of proposal distribution
  ai = 30 # adapt every ai iterations; set to iter+1 to turn off adaptation
  ari = 100 # base acceptance rate on this many iterations
  adapt.phase = 1000 # stop adapting after this many iterations
  accept[1,] = rep(1,p)
  scale = 1
  j = 2
  for(i in 2:iter){
   #adaptive
    if((i%%ai)==0 & i>1 & i<=adapt.phase){
      ap = pmax(0.01, apply(accept[max(1,i-ari):i,], 2, mean, na.rm=TRUE))
      scale = scale*(ap/0.44)
      cov <- scale*apply(beta, 2, sd, na.rm=TRUE)
    }
    b.prv = beta[i-1,]
    b.cand = b.prv
    for(q in 1:p){
     # update one at a time
     z = rep(0,p)
     z[q] = rnorm(1,0, cov[q])
     # non-normalized log-probability at previous beta
     #dbinom(y,1,expit(X%*%b.cand), log = TRUE)
     llp = sum(dbinom(y,1,expit(X%*%b.cand), log = TRUE)) + # log likelihood
       (dnorm(b.cand[1], mean = 0, sd = 10, log = TRUE)) + # 
       (dnorm(b.cand[2], mean = 0, sd = sqrt(0.5), log = TRUE)) # 
     #include draw from proposal dist'n
     b.cand = b.cand + z
     # non-normalized log-probability at new beta
     lp = sum(dbinom(y,1,expit(X%*%b.cand), log = TRUE)) + # log likelihood
       (dnorm(b.cand[1], mean = 0, sd = 10, log = TRUE)) + # 
       (dnorm(b.cand[2], mean = 0, sd = sqrt(0.5), log = TRUE)) # 
     l.rat = exp(lp-llp)
     a = rbinom(1, 1, min(1, l.rat))
     if(!a) b.cand = b.cand - z
     accept[i,q] = a
    }# loop over p
    beta[i,] = b.cand
  } #loop over i
  bt = data.frame(beta)
  names(bt) <- colnames(X)
  bt$iter = 1:dim(bt)[1]
  bt$chain = rep(chain, dim(bt)[1])
  list(beta = bt, accept=accept)
}

burn = 1000
samples = mh.adaptive.logistic(iter=41000, y=y, X=cbind(rep(1,length(x)), x), chain=1, prop.sigma.start=1, inits=NULL)
ests = samples$beta[samples$beta$iter>burn,]

ggplot(data = data.frame(ests)) + geom_line(aes(x=iter, y=x)) + theme_classic()
ggplot(data = data.frame(ests)) + geom_density(aes(x)) + theme_classic()

apply(ests[,1:2], 2, function(x) c(mean=mean(x), sd=sd(x),median=median(x),  or=exp(mean(x)), ll=exp(quantile(x, .025)), ll=exp(quantile(x, .975))))
# standard logistic model
summary(glm(y~x, family=binomial()))$coefficients


#### doing in MCMC software
# repeat in jags
jagsmod <- '
model{
  mu = X%*%b
 for(i in 1:N){
  y[i] ~ dbern(1/(1+exp(-mu[i])))
 } #N
 # priors
  b[1] ~ dnorm(0, 1/100)
  b[2] ~ dnorm(0, 1/(0.5))
} #model
'
tmpf2=tempfile()
tmpf2 = '/Users/akeil/temp/_____jmod.txt'
tmps=file(tmpf2,"w")
cat(jagsmod,file=tmps)
close(tmps)

burn = 1000
jags.data <- list(X = cbind(rep(1,length(x)), x),y=as.numeric(y),N=length(x))
jags.check <- jags.model(file=tmpf2, data=jags.data, n.chains=4, n.adapt = 1000)
update(jags.check, n.iter=burn)
jags.fit <- coda.samples(jags.check, variable.names=c('b[1]', 'b[2]'), n.iter=10000)
jags.sum <- summary(jags.fit)
b.jags  <- jags.sum$statistics[c('b[1]', 'b[2]'),1]
se.jags <- jags.sum$statistics[c('b[1]', 'b[2]'),2]
summary(jags.fit)
#traceplot(jags.fit)


# repeat in stan
stanmod <- '
data{
 int N;
 int<lower=0,upper=1> y[N];
 matrix[N,2] X;
}
parameters{
 vector[2] b;
}
model{
 b[1] ~ normal(0, 10);
 b[2] ~ normal(0, sqrt(0.5));
 y ~ bernoulli_logit(X*b);
} 
'
stan.data <- list(X = cbind(rep(1,length(x)), x),y=as.numeric(y),N=length(x))
stanmod.check <- stan(model_code = stanmod, data=stan.data, chains=1, iter=1, refresh=-1)
stanmod.fit <- stan(fit = stanmod.check, data=stan.data, chains=4,warmup = 1000, iter=11000, refresh=-1)
stanmod.fit
