library(ggplot2)
library(readr)
library(lasso2)
library(glmnet)
dat = read.table("https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt", header=TRUE)
names(dat) <- tolower(names(dat))
#dim(dat)
#write_csv(dat, "/Users/akeil/EpiProjects/mcmc/data/diabetes.csv")

head(dat)
summary(dat)

# least squares
X = as.matrix(subset(dat, select = names(dat)[1:10]))
#Xs = cbind(int=rep(1, dim(dat)[1]), (X-apply(X, 2, mean))/apply(X, 2, sd))
#y = (dat$y-mean(dat$y))/sd(dat$y)
Xs = cbind(int=rep(1, dim(dat)[1]), X)
y = dat$y
#ols
ols.coef = solve(t(Xs)%*%Xs)%*%t(Xs)%*%y
#lasso
lasso.coef = l1ce(y~., data=dat, bound=.5)$coef




mh.adaptive.diabetes <- function(iter, y, X=Xs, prop.sigma.start=1, inits=NULL) {
  #adaptive metropolis hastings
  p = dim(X)[2]
  accept <- beta <- matrix(nrow=iter, ncol=p)
  if(is.null(inits)) inits = runif(p)*4-2
  beta[1,] = inits
  cov =  rep(prop.sigma.start, p)# starting value for covariance parameter of proposal distribution
  ai = 50 # adapt every ai iterations
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
     sigma = sd(y-X%*%b.cand)
     llp = sum(dnorm(y, X%*%b.cand, sigma, log = TRUE)) + (q>1)*sum(dnorm(b.cand, log = TRUE)) # no prior on the intercept, standard normal otherwise
     #include draw from proposal dist'n
     b.cand = b.cand + z
     # non-normalized log-probability at new beta
     sigma = sd(y-X%*%b.cand)
     lp = sum(dnorm(y, X%*%b.cand, sigma, log = TRUE)) + (q>1)*sum(dnorm(b.cand, log = TRUE)) # no prior on the intercept, standard normal otherwise
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
  list(beta = bt, accept=accept)
}


post = mh.adaptive.diabetes(iter=30000, y=y, X=Xs, prop.sigma.start=0.001)
beta.post = post$beta
apply(post$accept, 2, mean)

burn = 0
suppressWarnings(if(burn==0) burn=-(1:dim(beta.post)[1]))
#names(dat) #  [1] "age" "sex" "bmi" "bp"  "s1"  "s2"  "s3"  "s4"  "s5"  "s6"  "y" 
ggplot(data=beta.post[-burn,]) + geom_line(aes(x=iter, y=int)) # intercept
ggplot(data=beta.post[-burn,]) + geom_line(aes(x=iter, y=age))
ggplot(data=beta.post[-burn,]) + geom_line(aes(x=iter, y=sex))
ggplot(data=beta.post[-burn,]) + geom_line(aes(x=iter, y=bmi))
ggplot(data=beta.post[-burn,]) + geom_line(aes(x=iter, y=bp))
ggplot(data=beta.post[-burn,]) + geom_line(aes(x=iter, y=s1)) # suspect
ggplot(data=beta.post[-burn,]) + geom_line(aes(x=iter, y=s2)) # suspect
ggplot(data=beta.post[-burn,]) + geom_line(aes(x=iter, y=s3)) # suspect
ggplot(data=beta.post[-burn,]) + geom_line(aes(x=iter, y=s4))
ggplot(data=beta.post[-burn,]) + geom_line(aes(x=iter, y=s5))
ggplot(data=beta.post[-burn,]) + geom_line(aes(x=iter, y=s6))

# posterior means, sds
t(apply(beta.post[-(1:2000), 1:11], 2, function(x) c(mean=mean(x), sd=sd(x))))
bayes.coef = apply(beta.post[-(1:2000), 1:11], 2, function(x) c(mean=mean(x)))
ols.coef
lasso.coef
summary(lm(y~., data=dat))
cor(X)
cor(beta.post[-(1:2000), 1:11])




horse_shoe_gibbs <- function(){
  TRUE
}



ggplot() + 
  geom_point(aes(x=ols.coef[-1], y=-(1:10), shape='OLS')) + 
  geom_point(aes(x=lasso.coef[-1], y=-(1:10), shape='LASSO (0.5)')) + 
  geom_point(aes(x=bayes.coef[-1], y=-(1:10), shape='Bayes')) + 
  scale_y_continuous(name='variable', breaks=-c(1:10), labels=1:10) + 
  scale_x_continuous(name=expression(beta[j])) + 
  scale_shape_discrete(name='') + 
  theme_classic() 
