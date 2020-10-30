# gibbs sampler for Bayesian horseshoe, logistic and linear model
# from A Simple Sampler for the Horseshoe Estimator
# Au: Enes Makalic and Daniel F. Schmidt

using LinearAlgebra, Distributions, DataFrames, StatsBase
using PolyaGammaDistribution
import wellwisejl.summarygibbs, wellwisejl.expit

# data
function expiti(mu::T) where T
  inv(1.0 + exp(-mu))
end

function expit!(mu::Array{T, 1}) where T
  mu .= expiti.(mu)
end

function meanexpit(mu::Array{T, 1}) where T
  mean(expit!(mu))
end


function initialize!(_beta, binits, p)
  if isnan(binits[1])
    _beta .= randn(p)*2
  else
    _beta .= copy(binits)
  end
end

function inputchecks(y,X,Xint,p, pl)
  if !any(isnan.(Xint[1]))
    for xx in Xint
      @assert size(xx,2)==size(X,2) "X and Xint have different numbers of columns"
    end
  end
  @assert size(y,1) == size(X,1) "y and X have different numbers of rows"
  @assert sum(pl) == p "sum(pl) should equal number of columns of X (without intercept)"
end

function inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, binits, pl=NaN)
  kappa = y .- 0.5
  # constants
  (N,p) = size(X)
  if isnan(pl[1])
    plidx, pls, j = NaN, NaN, NaN
  else
    j = size(pl, 1)
    inputchecks(y,X,Xint,p, pl)
    pls = vcat(2, (cumsum(pl) .+ 2)[1:(end-1)])
    plidx = [pls[g]:cumsum(pl)[g]+1 for g in 1:j]
  end
  Ni = size(Xint[1], 1)
  X = hcat(ones(N), X)
  _beta = Array{Float64,1}(undef, p+1)
  initialize!(_beta, binits, p+1)
  mu = X * _beta
  Xint = Xint
  Xt = transpose(X)
  xtx = Xt * X
  om = Array{Float64,1}(undef, N)
  Om = Diagonal(om)
  _A::Symmetric{Float64, Array{Float64, 2}} = Symmetric(Array{Float64}(undef, p+1, p+1))
  _A_ms = Array{Float64,1}(undef, p+1)
  _iLams = Diagonal(vcat(_tau_eta0^(-2), [_tau_beta0^(-2) for j in 1:p])) # precision prior on beta coefficients
  _muvec = vcat(_mu_eta0, [_mu_beta0 for j in 1:p])      # mean prior on beta coefficients
  munc, muint = Array{Float64,1}(undef, Ni), Array{Float64,1}(undef, Ni)
  #z = kappa ./ om
  _beta_store = Array{Float64,2}(undef, iter, p+1)
  _py_store = zeros(iter, 3)
  return(
    N, p, _beta, mu, kappa,
    pls, plidx, j, 
    X, Xint, Xt, xtx,
    om, Om,
    _A, _A_ms, _iLams, _muvec,
    munc, muint,
    _beta_store, _py_store
  )
end
inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, binits) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, binits, NaN)
inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, NaN, NaN)
inits(y,X,Xint,iter) = inits(y,X,Xint,iter,0.0,Inf,0.0,Inf, NaN, NaN)


function updateom!(rng, mu, om, Om, X, _beta)
  mu .= X * _beta
  om .= rand.(PolyaGamma.(1, mu))
  Om .= Diagonal(om)
end

function updatebeta!(rng, _beta, om, Om, X, Xt, kappa, _iLams, _muvec, _A, _A_ms)
   # block sampler for beta coefficients
   _A.data .= Xt * Om * X + _iLams
   _A_ms .= Xt * kappa .+ _iLams * _muvec   # for vector of precisions (iLams is diagonal of precision matrix)
  _beta .= rand(rng, MvNormalCanon(_A_ms, _A))
end

function updatebeta_or!(rng, _beta, om, Om, X, Xt, kappa, _iLams, _muvec, _A, _A_ms; K=11, p=p)
   # block sampler for beta coefficients
   # ordered over-relaxation by a random member of b
   # does not seem terribly effective
  _A.data .= Xt * Om * X + _iLams
  _A_ms .= Xt * kappa .+ _iLams * _muvec   # for vector of precisions (iLams is diagonal of precision matrix)
  orcand = rand(rng, MvNormalCanon(_A_ms, _A), K)
  sp = sortperm(vcat(_beta[bidx], orcand[bidx,:]))
  selidx = K - sp[1]  +2 
  _beta .= orcand[:,sp[2:end] .== selidx][:]
end




function calcmd!(munc, muint, Xint, _beta; offset=0.0)
  mul!(munc,  Xint[1], _beta[2:end])
  mul!(muint, Xint[2], _beta[2:end])
  munc .+= _beta[1] + offset
  muint .+= _beta[1] + offset
  cm1 = meanexpit(muint)
  cm0 = meanexpit(munc)
  md = cm1-cm0
  cm1, cm0, md
end

################################################################################################
#  Gibbs sampler based on Polya-gamma latent variables
################################################################################################


function gibbs_logit(y,X,Xint,iter,burnin,rng; binits=NaN, chain = 1.,
               offset = 0.0,
               _mu_eta0 = 0., _tau_eta0 = 1000.,                              # prior mean, sd of intercept
               _mu_beta0 = 0., _tau_beta0 = 100.0                                # prior mean, sd of beta
)
  (
    N, p, _beta, mu, kappa,
    pls, plidx, j, 
    X, Xint, Xt, xtx,
    om, Om,
    _A, _A_ms, _iLams, _muvec,
    munc, muint,
    _beta_store, _py_store
  ) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, binits)
  #
  for i in 1:iter
    # sample
    updateom!(rng, mu, om, Om, X, _beta)
    updatebeta!(rng, _beta, om, Om, X, Xt, kappa, _iLams, _muvec, _A, _A_ms)
    #updatebeta_or!(rng, _beta, om, Om, X, Xt, kappa, _iLams, _muvec, _A, _A_ms; K=11, p=p)
    if i > burnin  
      cm1, cm0, md = calcmd!(munc, muint, Xint, _beta, offset=offset)
      # store
      _beta_store[i,:] = _beta
      _py_store[i,:] = vcat(cm1, cm0, md)
    end
  end
  res = convert(DataFrame, hcat([chain for i in 1:iter], [i for i in 1:iter], _py_store, _beta_store))
  names!(res, vcat(:chain, :iter, :m1, :m0, :md, [Symbol("beta" * "[$i]") for i in 0:p]))#, [Symbol("rd" * "$i") for i in 1:nints]))
  return res[(burnin+1):iter,:]
end



################################################################################################
#  Hierarchical version (Gibbs)
################################################################################################

function updatemu!(rng, _mu, _muvec, _beta, j, pl, plidx, _tausq, _mu_tau0, _mu_mu0)    
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    iV = pl[q] / _tausq[q] +  _mu_tau0[q] .^(-2)
    MoV = (sum(bl) / _tausq[q] .+ _mu_mu0[q] * _mu_tau0[q] .^(-2))
    _mu[q] = rand(rng, NormalCanon(MoV, iV))
    _muvec[plidx[q]] .= _mu[q]
  end
end

function updatetausq!(rng, _tausq, _iLams, _beta, j, pl, plidx, _mu, _tau_mu0, _tau_tau0)
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    bres = bl .- _mu[q]
    bse = transpose(bres) * bres
    a_tau = _tau_mu0[q] + pl[q]/2.
    b_tau = _tau_tau0[q] + bse/2.
    _tausq[q] = rand(rng, Gamma(a_tau, 1. / b_tau))
    _iLams.diag[plidx[q]] .= inv(_tausq[q])
  end
end

function gibbs_hlogit(y,X,Xint,iter,burnin,rng; binits=NaN,
               pl = [size(X,2)],
               offset = 0.0, chain = 1,
               _mu_eta0 = 0., _tau_eta0 = 1000.,                              # prior mean, sd of intercept
               _mu_beta0 = 0., _tau_beta0 = 100.0,                                # prior mean, sd of beta
               _tau_mu0 = 0., _tau_tau0 = 1.0,                                # prior mean, sd of beta
               _mu_mu0= 0.0, _mu_tau0 = 10.0
)
  (
    N, p, _beta, mu, kappa,
    pls, plidx, j, 
    X, Xint, Xt, xtx,
    om, Om,
    _A, _A_ms, _iLams, _muvec,
    munc, muint,
    _beta_store, _py_store
  ) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, binits, pl)
  #
  _mu = randn(rng, j)
  _tausq = rand(rng, j)
  _mu_store = Array{Float64, 2}(undef, iter, j)
  _tausq_store = Array{Float64, 2}(undef, iter, j)
  for i in 1:iter
    # sample
    updateom!(rng, mu, om, Om, X, _beta)
    updatebeta!(rng, _beta, om, Om, X, Xt, kappa, _iLams, _muvec, _A, _A_ms)
    #updatebeta_or!(rng, _beta, om, Om, X, Xt, kappa, _iLams, _muvec, _A, _A_ms; K=11, p=p)
    updatemu!(rng, _mu, _muvec, _beta, j, pl, plidx, _tausq, _mu_tau0, _mu_mu0)    
    updatetausq!(rng, _tausq, _iLams, _beta, j, pl, plidx, _mu, _tau_mu0, _tau_tau0)
    if i > burnin  
      cm1, cm0, md = calcmd!(munc, muint, Xint, _beta, offset=offset)
      # store
      _beta_store[i,:] = _beta
      _mu_store[i,:] = _mu
      _tausq_store[i,:] = _tausq
      _py_store[i,:] = vcat(cm1, cm0, md)
    end
  end
  rr = hcat([chain for i in 1:iter], [i for i in 1:iter], 
   _py_store, _beta_store, _mu_store, _tausq_store)
  nms = vcat(:chain, :iter, :m1, :m0, :md, [Symbol("beta" * "[$i]") for i in 0:p], [Symbol("mu" * "[$i]") for i in 1:j], [Symbol("tausq" * "[$i]") for i in 1:j])
  res = convert(DataFrame, rr)
  names!(res, nms)#, [Symbol("rd" * "$i") for i in 1:nints]))
  return res[(burnin+1):iter,:]
end

################################################################################################
#  Slice sampler
################################################################################################
function logitbin(_beta, X, y, _mu_eta0 = 0., _tau_eta0 = 1000., _mu_beta0 = 0., _tau_beta0 = 100.0)
  lmu = -log1p.(exp.(.- X * _beta))
  l1mmu = -log1p.(exp.(X * _beta))
  ll = lmu .* y .+ l1mmu .* (1 .- y)
  ret = sum(ll)
  if isfinite(_tau_eta0)
    ret += logpdf(Normal(_mu_eta0,_tau_eta0), _beta[1])
  end
  if isfinite(_tau_beta0)
    ret += sum(logpdf.(Normal(_mu_beta0,_tau_beta0),  _beta[2:end]))
  end
  ret
end

function stepleft!(tsteps, lpostfun, btem, w, j,  X, y, _mu_eta0, _tau_eta0, _mu_beta0, _tau_beta0, loght)
  while true
    btem[j] -= w[j]
    fxcand = lpostfun(btem, X, y, _mu_eta0, _tau_eta0, _mu_beta0, _tau_beta0)
    fxcand < loght && break
    if isnan(fxcand) 
      btem[j] += w[j]      
      break
    end
    tsteps[1] += 1
  end
end

function stepright!(tsteps, lpostfun, btem, w, j,  X, y, _mu_eta0, _tau_eta0, _mu_beta0, _tau_beta0, loght)
  while true
    btem[j] += w[j]
    fxcand = lpostfun(btem, X, y, _mu_eta0, _tau_eta0, _mu_beta0, _tau_beta0)
    fxcand < loght && break
    if isnan(fxcand) 
      btem[j] -= w[j]      
      break
    end
    tsteps[1] += 1
  end
end

function adaptweights!(rng, w, j, p, tsteps; ul=12.0, ll=3.0/2.0)
  if tsteps[1] > p*ul
    w[j] *= (1+rand(rng))
  elseif tsteps[1] < p*ll
    w[j] *= rand(rng)            
  end
end

function rejectstep!(rng, lpostfun, loght, btem, _beta, ap, j, bu, bl, X, y, _mu_eta0, _tau_eta0, _mu_beta0, _tau_beta0)
  ob = _beta[j]
  nsamps = 1.0
  while true
    btem[j] = _beta[j]
    # sample point between two pdfs 
    _beta[j] = rand(rng) * (bu[1] - bl[1]) + bl[1]
    # small number of rejections here to deal with the fact that we may have sampled slightly outside the pdf
    # if beta has higher pdf than bounds of the horizontal lines, then move bounds inward and repeat
    fxcand = lpostfun(_beta, X, y, _mu_eta0, _tau_eta0, _mu_beta0, _tau_beta0)
    #if new candidate is greater than old one, we missed on the right: move bu to new spot
    fxcand > loght && break
    if fxcand < loght && _beta[j] > ob
      bu[1] = _beta[j]
      nsamps += 1.0
    elseif fxcand < loght && _beta[j] <= ob
      bl[1] = _beta[j]
      nsamps += 1.0
    end
  end
  ap[j] += inv(nsamps)
end

function slice_logit(y,X,Xint,iter,burnin,rng; 
               chain = 1,
               lpostfun=logitbin,binits=NaN,
               offset = 0.0,
               _mu_eta0 = 0., _tau_eta0 = 1000.,                              # prior mean, sd of intercept
               _mu_beta0 = 0., _tau_beta0 = 100.0,
               winit=0.1
               )
  (
    N, p, _beta, mu, kappa,
    pls, plidx, j, 
    X, Xint, Xt, xtx,
    om, Om,
    _A, _A_ms, _iLams, _muvec,
    munc, muint,
    _beta_store, _py_store
  ) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, binits)

  w = fill(winit, p+1)
  __lp = Array{Float64, 1}(undef, iter)
  #
  btem = copy(_beta)
  ap = zeros(p+1)
  bu,bl = ones(1), ones(1)
  for i in 1:iter
    ord = randperm(rng, size(_beta,1)) # go through variables in random order
    for j in ord
      ####################################################################################################
      # sample a latent point directly under the pdf of the current parameter value (beta_old)
      ####################################################################################################
      lp = lpostfun(_beta, X, y, _mu_eta0, _tau_eta0, _mu_beta0, _tau_beta0)
      # sample from the vertical line under the conditional density at x
      loght = copy(lp)
      loght += log(rand(rng))
      ####################################################################################################
      # find parameter values that correspond to the same pdf values as the latent point
      ####################################################################################################
      #expand left bounds until point is outside of slice (log pdf at bound is lower than current point)
      tsteps = [0]
      btem .= _beta
      stepright!(tsteps, lpostfun, btem, w, j,  X, y, _mu_eta0, _tau_eta0, _mu_beta0, _tau_beta0, loght)
      bu[1] = btem[j]
      #expand right bounds until point is outside of slice (log pdf at bound is lower than current point)
      btem[j] = _beta[j]
      stepleft!(tsteps, lpostfun, btem, w, j,  X, y, _mu_eta0, _tau_eta0, _mu_beta0, _tau_beta0, loght)
      bl[1] = btem[j]
      # adaptive step size
      ####################################################################################################
      # adapt weights for more efficient bounding of pdf
      ####################################################################################################
      #if i < burnin
        adaptweights!(rng, w, j, p, tsteps, ul=8.0, ll=1.4)
      #end
      ####################################################################################################
      # Metropolis step: sample new point (beta_new) between the bounds, reject/accept with probability a
      #  where acceptance probability is the ratio of pdfs at the current point and at the previous
      #  point (beta_old)
      ####################################################################################################
      # rejection sample from the horizontal line: if sample has higher probability, accept, else accept with probability proportional to posterior ratio
      rejectstep!(rng, lpostfun, loght, btem, _beta, ap, j, bu, bl, X, y, _mu_eta0, _tau_eta0, _mu_beta0, _tau_beta0)
    end
    if i > burnin  
      cm1, cm0, md = calcmd!(munc, muint, Xint, _beta, offset=offset)
      _beta_store[i,:] = _beta
      _py_store[i,:] = vcat(cm1, cm0, md)
      __lp[i] = lpostfun(_beta, X, y, _mu_eta0, _tau_eta0, _mu_beta0, _tau_beta0)
    end
  end
  println("Acceptance ratio")
  println(ap/iter)
  res = DataFrame(hcat([chain for i in 1:iter], [i for i in 1:iter], _py_store, _beta_store,__lp))
  names!(res, vcat(:chain, :iter, :m1, :m0, :md, [Symbol("beta" * "[$i]") for i in 0:p], :__lp))#, [Symbol("rd" * "$i") for i in 1:nints]))
  return res[(burnin+1):iter,:]
end
;