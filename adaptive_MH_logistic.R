library(foreach)
library(doParallel)
library(rjags)
library(coda)

expit <- function(mu) 1/(1+exp(-mu))
dgm <- function(n){
  z = rnorm(n)
  x = rbinom(n,1,expit(z))
  y = rbinom(n,1,expit(z))
  int=rep(1,n)
  tibble(int=int, x=x,y=y,z=z)
}




mh.adaptive.logistic <- function(iter, y, X=Xs, chain=1, prop.sigma.start=1, inits=NULL) {
  #adaptive metropolis hastings
  p = dim(X)[2]
  accept <- beta <- matrix(nrow=iter, ncol=p)
  if(is.null(inits)) inits = runif(p)*4-2
  beta[1,] = inits
  cov =  rep(prop.sigma.start, p)# starting value for covariance parameter of proposal distribution
  ai = 10 # adapt every ai iterations; set to iter+1 to turn off adaptation
  ari = 50 # base acceptance rate on this many iterations
  adapt.phase = Inf # stop adapting after this many iterations
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
       sum(dnorm(b.cand, mean = 0, sd = 1, log = TRUE)) # log standard normal priors
     #include draw from proposal dist'n
     b.cand = b.cand + z
     # non-normalized log-probability at new beta
     lp = sum(dbinom(y,1,expit(X%*%b.cand), log = TRUE)) + # log likelihood
       sum(dnorm(b.cand, mean = 0, sd = 1, log = TRUE)) # log standard normal priors
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

dat <- dgm(100)
y = dat$y
Xs = as.matrix(dat[,c('int','x','z')])

beta.post <- foreach(i=1:4, .combine='rbind', .export=c('mh.adaptive.logistic')) %dopar% {
  mh.adaptive.logistic(iter=10000, chain=i, y=y, X=Xs, prop.sigma.start=0.001)$beta
}

# acceptance probability
apply(post$accept, 2, mean)
  
burn = 0
#names(dat) #  [1] "age" "sex" "bmi" "bp"  "s1"  "s2"  "s3"  "s4"  "s5"  "s6"  "y" 
ggplot(data=beta.post[beta.post$iter>burn,]) + geom_line(aes(x=iter, y=int, color=factor(chain))) # intercept
ggplot(data=beta.post[beta.post$iter>burn,]) + geom_line(aes(x=iter, y=x, color=factor(chain))) 
ggplot(data=beta.post[beta.post$iter>burn,]) + geom_line(aes(x=iter, y=z, color=factor(chain))) 


# repeat in jags
jagsmod <- '
model{
 for(i in 1:N){
  mu[i] = b0 + t(X[i,])%*%b
  y[i] ~ dbern(1/(1+exp(-(mu[i]))))
 } #N
 # priors
 for(j in 1:P){
  b[j] ~ dnorm(0, 1)
 }
 b0 ~ dnorm(0, 1/(10^2)) # vague (cant leave unspecified in jags)
} #model
'
tmpf2=tempfile()
tmpf2 = '/Users/akeil/temp/_____jmod.txt'
tmps=file(tmpf2,"w")
cat(jagsmod,file=tmps)
close(tmps)

burn = 500
jags.data <- list(X = as.matrix(dat[,c('x','z')]),y=as.numeric(dat$y),N=dim(dat)[1],P=2)
jags.check <- jags.model(file=tmpf2, data=jags.data, n.chains=4)
update(jags.check, n.iter=burn)
jags.fit <- coda.samples(jags.check, variable.names=c('b0', 'b'), n.iter=10000)
jags.sum <- summary(jags.fit)
b.jags <- jags.sum$statistics[c('b0', 'b[1]', 'b[2]'),1]
se.jags <- jags.sum$statistics[c('b0', 'b[1]', 'b[2]'),2]

traceplot(jags.fit)

#compare with standard logistic, jags
summary(glm(y ~ x + z, family=binomial(), data=dat))$coefficients[,1:2]
t(apply(beta.post[beta.post$iter>burn,1:3], 2, function(x) c(mean=mean(x), sd=sd(x))))
cbind(mean=b.jags, sd =se.jags)



