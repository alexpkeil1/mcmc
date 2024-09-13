"GIBBS sampling for latent class analysis based on a series of binary indicators"
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6364555/
# GIBBS lca

# class: c
# pjk = probability that person in jth latent group endorses the kth symptom

"""
library(poLCA)
ah = read.csv("/Users/keilap/Documents/DataSets/PublicData/AddHealth/ICPSR_21600/DS0001/addhealth_lca.csv")

ah = ah+1

poLCA(cbind(lied, publicly, damaged, shoplift, stole50, grpfight)~1, data=ah, nclass=4)








gibbs <- function(y, G, dirich.prior = NULL, niter = 7500,
                  n.burn = 2500, n.thin = 10, relabel = TRUE, verbatim = TRUE) {
  if ( ! all(y == 0 | y == 1) )
    stop("y must be coded 0 or 1")      # stop if y is not coded 0, 1
  if ( niter <= n.burn )				# niter has to be > n.burn, error if not
    stop(paste("niter =", niter, ", must be greater than n.burn =", n.burn))
  ###
  # loading packages needed to run Gibbs sampling and basic settings
  ###
  require(gtools)	# rdirichlet()
  K <- ncol(y)	# number of items
  N <- nrow(y)	# number of respondents
  G <- G	# number of latent groups
  done.burn <- FALSE # burn.in is not yet done
  ###
  # MCMC basic setup, number of iterations and storages for chains
  ###
  Pi <- matrix(NA, nrow = niter, ncol = G) # storage for class membership
  Pjk <- array(NA, dim = c(niter, G, K))	# storage for item resp prob
  dimnames(Pjk) <- list(NULL, paste("G", 1:G, sep = ""), colnames(y))
  Cij <- array(NA, dim = c(niter, N, G))	# storage for discrete classes
  Cij.pr <- array(NA, dim = c(niter, N, G)) # storage for latent class prob
  labelStor <- matrix(NA, nrow = niter, ncol = G) # storage for relabeling
  #
  ## Storages for simulated parameters pjk, C, at iteration t+1
  #
  pjk.t1 <- matrix(NA, nrow = G, ncol = K) # latest p_jk^(t+1) stored here
  dimnames(pjk.t1) <- list(paste("G", 1:G, sep=""), colnames(y))
  # N*G (people by group) matrix of each person’s class membership prob
  Clp.t1 <- matrix(NA, nrow = N, ncol = G)
  dimnames(Clp.t1) <- list(paste("N", 1:N, sep=""), paste("G", 1:G, sep=""))
  ###
  # Priors
  ###
  if ( is.null(dirich.prior) )
    dirich.prior <- rep(1, G) # flat Dirichlet by default
  # Beta prior, alpha=1 and beta=1 for a flat prior
  alpha <- 1
  beta <- 1
  ###
  # Starting values of pi and pjk, drawn randomly from Dirichlet, Beta priors
  ###
  start.pi <- rdirichlet(n=1, alpha = dirich.prior)
  start.item.p <- matrix(NA, nrow = G, ncol = K)
  for (g in 1:G)
  {
    start.item.p[g, ] <-
      rbeta(K, shape1 = alpha, shape2 = beta)
  }
  ###
  pi.t <- start.pi # membership distr [pi1=0.78, pi2=0.11, pi3=0.11]
  pjk.t <- start.item.p # item response probability pjk on iteration t
  # used later to address the label switch problem
  perm <- gtools::permutations(n=G, r=G) # 24 total permutations when G=4
  trace.num <- numeric(nrow(perm))	# trace of match between t0 and t+1 55 ################
  # Main MCMC simulation
  ################
  iter <- 1			# begins with iteration number 1
  while (iter <= niter)		# loop until niter is reached
  {
    # Each person’s class membership prob, using Eq (7)
    #	[c|y,pi,p] = [pi * pr(y|p)] / [total probability]
    # step 1: pr(y|p), first calcualte p and 1-p
    pr.p <- t(pjk.t)	# transpose to K by G matrix for apply()
    pr.q <- 1 - pr.p
    # step 2: binomial item response probability per Eq (2)
    A <- apply(y, MAR = 1, FUN = function(yv) { pr.p^yv * pr.q^(1-yv) })
    A <- array(A, dim = c(K, G, N))
    A <- aperm(A, c(3, 2, 1))	# reshape into N*G*K
    eq2 <- apply(A, MARGIN = c(1, 2), prod)  # multiply across K, keeping N*G
    # step 3: each binomial item resp prob weighted by class distr prob pi[j]
    eq2 <- sweep(eq2, MARGIN = 2, STATS = pi.t, FUN = "*")
    # Calculate total probability for each person, per Eq (5)
    p.total <- apply(eq2, MARGIN = 1, sum)
    # finally, ‘divided-by-total’ yields latent class membership prob
    Clp.t1 <- eq2/p.total
    #
    # Clp.t1 gives us the probability of each person’s latent class membership,
    # e.g., person 1 has (0.30, 0.20, 0.15, 0.35) of being in class 1, 2, 3, and 4.
    # So latest class membership can be c=[1,0,0,0] with 30% chance,
    # c=[0,1,0,0] with 20% chance, c=[0,0,1,0] with 15% chance, and c=[0,0,0,1]
    # with 35% chance. Next we use these probs to draw a single specific sample
    # of c from any of the 4 possibilities above. Each person has one and only
    # one class out of G latent classes.
    Cl.t1 <- apply(Clp.t1, 1,
                   function(prob) { rmultinom(n = 1, size = 1, prob = prob) })
    Cl.t1 <- t(Cl.t1)
    ##
    # Next, update pi (per Eq (10)) and pjk (per Eq (11)) using the newly
    # calculated N*G matrix of discrete latent class membership
    ##
    # Sample pi_j^{(t+1)}, percentages of latent classes in the population
    # Eq (10) shows posterior = data by colSums(C.t) + prior sample sizes
    pi.t1 <- rdirichlet(n = 1, alpha = colSums(Cl.t1) + dirich.prior)
    # sample item response probability, one latent class at a time, sum over N
    for (g in 1:G) # each column not guaranteed to add up to 1
    {
      # Eq (11) shows posterior beta(y*c + alpha, (1-y)*c + beta)
      pjk.t1[g, ] <- rbeta(K, shape1 = alpha + colSums(Cl.t1[, g] * y),
                           shape2 = beta + colSums(Cl.t1[, g] * (1-y)) )
    }
    # simulated values in current iteration are added into chain storages
    Pi[iter, ] <- pi.t1
    Pjk[iter, , ] <- pjk.t1
    Cij[iter, , ] <- Cl.t1
    Cij.pr[iter, , ] <- Clp.t1
    # ‘label switching’ problem to match latent classes at end of burn-in
    if (relabel && done.burn)
    {
      match.tab <- t(Cl.t1) %*% Cl.0 # match between t+1 and t0 latent classes
      for (l in 1:nrow(perm)) # across G! permutations, where matches are?
        trace.num[l] <- sum(diag(match.tab[, perm[l, ]]))
      
      relabel.ord <- perm[which.max(trace.num), ] # relabel by best match
      labelStor[iter, ] <- relabel.ord
    }
    # Current simulated values will be used to draw the next iteration
    pi.t <- pi.t1
    pjk.t <- pjk.t1
    # print a message if b.burn iterations done
    if (iter == n.burn) {
      done.burn <- TRUE
      cat("\nburn-in completed\n")
      Cl.0 <- Cl.t1 # latent classes immediately after burn-in
    }
    # verbatim can be set by the user to print iteration count every 500th iter
    if (verbatim)
      if (iter == 1) cat("iteration(s) completed: ", iter, " ")
    if ( (iter %% 500) < 10^(-7) ) { cat(iter, " ") }
    iter <- iter + 1 # last thing before repeating is to incrememt iter by 1
  }	# end while (iter <= niter)
  cat("\n")
  # Discard burn-in iterations, thin by n.thin
  ti <- seq(from = n.burn+1, to = niter, by = n.thin)
  Pi.chain <- Pi[ti, ]			# pi chain after burn-in and thinning
  Pjk.chain <- Pjk[ti, ,]		# pjk after burn-in and thinning
  labelStor <- labelStor[ti, ]
  Pi.mean <- apply(Pi.chain, 2, mean) # average pi
  Pjk.mean <- apply(Pjk.chain, c(2, 3), mean)  # average p[jk]
  # put the results together in a list and return() the results
  ans <- list(Pi.mean = Pi.mean, Pjk.mean = Pjk.mean, Pi.chain = Pi.chain,
              Pjk.chain = Pjk.chain, Cpr.chain = Cij.pr, relabelOrd = labelStor)
  return(ans)
}

y = as.matrix(ah) - 1
gibbs(y, G = 4, niter = 12000)

"""

using Distributions, Random, Combinatorics, LinearAlgebra

function pi_i(classvec, u)
    classvec
end

"""
# posterior (marginal) probability of being in a class
p(C)

classvecs = [
    1 0 0;
    0 1 0;
    0 1 0;
    1 0 0;
    1 0 0;
    1 0 0;
    1 0 0;
    1 0 0;
    0 0 1 
]
pi = ones(3)
u = ones(3)
sample_pi!(pi, classvecs, u)
"""
function sample_pi!(pi, classvecs, u)
    #pli = [classvecs[i,:] for i in 1:size(classvecs,1)]
    pl = sum(classvecs, dims=1)[:] + u
    pi .= rand(Dirichlet(pl))
    nothing
end


"""
Posterior probability vector for measured variables, given class
P(Y | C)

Each Y follows a beta distribution within each class

classvecs = [
    1 0 0;
    0 1 0;
    0 1 0;
    1 0 0;
    1 0 0;
    1 0 0;
    1 0 0;
    1 0 0;
    0 0 1 
]
n = 9
C = 4 # class index
K = size(y,2) # unique binary exposures index
#y = Int.(rand(n,K) .> 0.5)
yi = y[1,:]
ajk = ones(C,K)
bjk = ones(C,K)
pjk = ones(C,K)
sample_pjk!(pjk, C, y, classvecs, ajk, bjk)
"""
function sample_pjk!(rng, pjk, C, y, classvecs, ajk, bjk)
    for c in 1:C
        a = classvecs[:,c:c]'*y .+ ajk[c:c,:]# .- 1
        b = classvecs[:,c:c]'*(1.0 .- y) .+ bjk[c:c,:]# .- 1
        pjk[c,:] .= rand.(rng, Beta.(a[:],b[:]))
    end
    nothing
end

sample_pjk!(pjk, J, y, classvecs, ajk, bjk) = sample_pjk!(MersenneTwister(), pjk, J, y, classvecs, ajk, bjk)


function classvecs_i(C, pi_, yi, pjk)
    # p(c) * p(y|c) / p(y)
    cv = zeros(C)
    for c in 1:C
      lpj = yi .* log.(pjk[c,:]) .+  (1.0 .- yi) .* log.(1.0 .- pjk[c,:])
      cv[c] += exp(log(pi_[c])+sum(lpj)) # cv[j] = p(j) * p(y|j)
    end
    # pi_' * pjk
    cv ./ sum(cv)
end

"""
Posterior draws of classes, given the outcome
pi_ = [0.2792204, 0.1283047, 0.07102063, 0.5214543]
pjk=[
0.3072238 0.5514973 0.3462326 0.03370501 0.5089951 0.3452769;
0.5733021 0.6741166 0.5370192 0.60961181 0.4667233 0.3464321;
0.9712629 0.2293985 0.5746601 0.77493576 0.4271499 0.7361765;
0.8057957 0.1349706 0.1792205 0.65392391 0.6252457 0.4477678
]

classvecs_i(C, pi_, y[1,:], pjk)

P(C|Y) = (P(C)*P(Y|C))/(SUM P(Y|C))
"""
function sample_classvecs!(rng, classvecs, pi_, y, pjk, C)
    for i in 1:size(y,1)
        classvecs[i,:] .= rand(rng, Multinomial(1, classvecs_i(C, pi_, y[i,:], pjk)))
    end
    nothing
end
sample_classvecs!(classvecs, pi, y, pjk, C) = sample_classvecs!(MersenneTwister(), classvecs, pi, y, pjk, C)

function labswitch_classmove(cvi, cvcur, perms)
    matchmat = cvi' * cvcur #.= classvecs[:,sp]
    whichp = argmax([sum(diag(matchmat[:,p])) for p in perms])
    perms[whichp]
end



"""
labs = post["labels"]
parm = post["pi"]
mean(parm, dims=1)
relabelpost!(parm, labs, 1)
mean(post["pi"], dims=1)
mean(parm, dims=1)
"""
function relabelpost!(parm, labs, dims=1)
  for row in 1:size(parm,1)
    tmp = parm[row,labs[row,:]]
    parm[row,:] .= tmp
  end
  nothing
end


"""
# fully worked example
#using Add Health data linked in: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6364555/
using CSV, DataFrames
nmorig = [:AID,:H1DS3,:H1DS15,:H1DS2,:H1DS4,:H1DS13,:H1DS14]
nmnew =  [:AID,:lied,:publicly,:damaged,:shoplift,:stole50,:grpfight]
addhealth = CSV.read("/Users/keilap/Documents/DataSets/PublicData/AddHealth/ICPSR_21600/DS0001/21600-0001-Data.tsv", DataFrame, delim="\t", select=nmorig)[:,nmorig]
rename!(addhealth, nmnew)

y = Matrix(Int.(addhealth[:,2:end] .> 0))

post = gibbs_lca(y,4;iters=12000, burnin=2500)

extrema(post["labels"], dims=1) # check for label switching

mean(post["pi"], dims=1)
# example of re-labeling to address label switching
#relabelpost!(post["pi"], labs, 1)
#mean(post["pi"], dims=1)

# re-creating posterior mean table
DataFrame(hcat(vcat(:pi_j,nmnew[2:end]), round.(vcat(mean(post["pi"], dims=1),
     reshape(mean(post["pjk"], dims=1), (4,6))'), digits=3)), [:_, :c1, :c2, :c3, :c4])

# Trace plot example
using Plots
plot(post["pi"][100:end,1])

plot(post["pjk"][100:end,7])
"""
function gibbs_lca(rng, y,C;iters=100, burnin=0)
    # pre-reqs
    N,P = size(y)
    perms = [p for p in permutations(collect(1:C))] # label switching
    nperms = factorial(4)
    # priors (to be moved into function)
    u = ones(C)
    ajk = ones(C,P)
    bjk = ones(C,P)
    # end priors (to be moved into function)
    # initialize
    classlabs = sample(rng, 1:C, N)
    classvecs = zeros(Int64,N,C)
    classvecs_init = zeros(Int64,N,C)
    for i in 1:N
      classvecs[i,classlabs[i]] = 1
    end
    pi_ = rand(rng, C)
    pjk = rand(rng, C,P)
    # containers
    pi_post = Array{Float64,2}(undef, iters-burnin, C)
    pjk_post = Array{Float64,2}(undef, iters-burnin, C*P)
    class_post = Array{Float64,2}(undef, iters-burnin, N)
    reord_post = Array{Int64,2}(undef, iters-burnin, C)
    if burnin == 0
      pi_post[1,:] = pi_
      pjk_post[1,:] = pjk[:] # down in columns: p[1,1], p[2,1], ... etc.
      class_post[1,:] = [c[2] for c in argmax(classvecs, dims=2)] # down in columns: classvec[1,1], classvec[2,1], ... etc.
    end
    # iterate
    for sweep in 2:iters
        sample_classvecs!(MersenneTwister(), classvecs, pi_, y, pjk, C)
        sample_pi!(pi_, classvecs, u)
        sample_pjk!(MersenneTwister(), pjk, C, y, classvecs, ajk, bjk)
        # 
        if (sweep == 2 && burnin==0) || (sweep == burnin)
            classvecs_init .= deepcopy(classvecs)
        end
        if sweep > burnin
          # label switching (which permutation of the confusion matrix columns gives the largest diagonal?)
          ord = labswitch_classmove(classvecs_init, classvecs, perms)
          #pi .= pi[ord]
          #pjk .= pjk[ord,:]
          #classvecs .= classvecs[:,ord]
          
          pi_post[sweep-burnin,:] .= pi_
          pjk_post[sweep-burnin,:] .= pjk[:] # down in columns: p[1,1], p[2,1], ... etc.
          class_post[sweep-burnin,:] .= [c[2] for c in argmax(classvecs, dims=2)] # down in columns: classvec[1,1], classvec[2,1], ... etc.    
          reord_post[sweep-burnin,:] .= ord # down in columns: classvec[1,1], classvec[2,1], ... etc.    
        end
    end
    Dict(
        "pi" => pi_post, 
        "pjk" => pjk_post, 
        "class" => class_post,
        "labels" => reord_post
    )
end
gibbs_lca(y,C;kwargs...) = gibbs_lca(MersenneTwister(), y,C;kwargs...)

