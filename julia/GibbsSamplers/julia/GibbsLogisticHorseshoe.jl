# gibbs sampler for Bayesian horseshoe, logistic and linear model
# from A Simple Sampler for the Horseshoe Estimator
# Au: Enes Makalic and Daniel F. Schmidt
using LinearAlgebra, Distributions, DataFrames, StatsBase
using PolyaGammaDistribution
import wellwisejl.summarygibbs

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

function inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0,binits)
  kappa = y .- 0.5
  # constants
  (N,p) = size(X)
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
  _iLams = vcat(_tau_eta0^(-2), [_tau_beta0^(-2) for j in 1:p]) # precision prior on beta coefficients
  _muvec = vcat(_mu_eta0, [_mu_beta0 for j in 1:p])      # mean prior on beta coefficients
  munc, muint = Array{Float64,1}(undef, Ni), Array{Float64,1}(undef, Ni)
  #z = kappa ./ om
  _beta_store = Array{Float64,2}(undef, iter, p+1)
  _py_store = zeros(iter, 3)
  return(
    N, p, _beta, mu, kappa,
    X, Xint, Xt, xtx,
    om, Om,
    _A, _A_ms, _iLams, _muvec,
    munc, muint,
    _beta_store, _py_store
  )
end

inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, NaN)
inits(y,X,Xint,iter) = inits(y,X,Xint,iter,0.0,Inf,0.0,Inf, NaN)

function updateom!(rng, mu, om, Om, X, _beta)
  mu .= X * _beta
  om .= rand.(PolyaGamma.(1, mu))
  Om .= Diagonal(om)
end

function updatenu!(_nu, _lamsq, p)
  # local shrinkage latent variable (p length)
  na = 1.0
  nb = 1.0 .+ inv.(_lamsq)
  _nu .= rand.(rng, InverseGamma.(na,nb))
end

function updatelamsq!(_lamsq, _tausq, _sigsq, _nu, _beta, p)
  # local shrinkage parameter (p length)
  la = 1.0
  lb = inv.(_nu) 
  lb .+= _beta .^2.0 ./ (2.0 .* _tausq .* _sigsq )
  _lamsq .= rand.(rng, InverseGamma.(la,lb))
end

function samplezeta(_tausq)
  # global shrinkage latent variable
  za = 1.0
  zb = 1.0 + inv(_tausq)
  rand(rng, InverseGamma( za, zb ))
end

function sampletausq(_sigsq, _lamsq, _zeta, _beta, p)
  # global shrinkage parameter
  ta = ( p + 1.0 ) / 2.0
  tb = inv(_zeta) 
  tb += inv( 2.0 * _sigsq ) * sum(_beta .^2.0 ./ _lamsq )
  rand(rng, InverseGamma(ta, tb))
end

function updateiLams!(_iLams, _tausq, _lamsq)
  _iLams[2:end] .= inv.(_tausq .* _lamsq)
end

function updatebeta!(rng, _beta, om, Om, X, Xt, kappa, _iLams, _muvec, _A, _A_ms)
   # block sampler for beta coefficients
   _A.data .= Xt * Om * X + Diagonal(_iLams)
   _A_ms .= Xt * kappa .+ _iLams .* _muvec   # for vector of precisions (iLams is diagonal of precision matrix)
  _beta .= rand(rng, MvNormalCanon(_A_ms, _A))
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

function gibbs_horselogit(y,X,Xint,iter,burnin,rng; binits=NaN,
               offset = 0.0,
               _mu_eta0 = 0., _tau_eta0 = 1000.,                              # prior mean, sd of intercept
               _mu_beta0 = 0., _tau_beta0 = 100.0                                # prior mean, sd (ignored) of beta
)
  (
    N, p, _beta, mu, kappa,
    X, Xint, Xt, xtx,
    om, Om,
    _A, _A_ms, _iLams, _muvec,
    munc, muint,
    _beta_store, _py_store
  ) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0,binits)
  #
  _nu = rand(rng, p)
  _lamsq = rand(rng, p)*2.0
  _zeta = rand(rng)
  _tausq = rand(rng)*2.0
  _tausq_store = Array{Float64, 1}(undef, iter)
  _lamsq_store = Array{Float64, 2}(undef, iter, p)
  for i in 1:iter
    # sample
    ##############################
    # update global shrinkage
    ##############################
    updatenu!(_nu, _lamsq, p)
    updatelamsq!(_lamsq, _tausq, 1.0, _nu, _beta[2:end], p)
    ##############################
    # update local shrinkage
    ##############################
    _zeta = samplezeta(_tausq)
    _tausq = sampletausq(1.0, _lamsq, _zeta, _beta[2:end], p)
    updateiLams!(_iLams, _tausq, _lamsq)
    ##############################
    # update model parameters
    ##############################
    updateom!(rng, mu, om, Om, X, _beta)
    updatebeta!(rng, _beta, om, Om, X, Xt, kappa, _iLams, _muvec, _A, _A_ms)
    ##############################
    # update summary parameters
    ##############################
    if i > burnin  
      cm1, cm0, md = calcmd!(munc, muint, Xint, _beta, offset=offset)
      ##############################
      # storage
      ##############################
      _tausq_store[i] = _tausq
      _lamsq_store[i,:] = _lamsq
      _beta_store[i,:] = _beta
      _py_store[i,:] = vcat(cm1, cm0, md)
    end
  end
  res = convert(DataFrame, hcat([chain for i in 1:iter], [i for i in 1:iter], 
   _py_store, _beta_store, _lamsq_store, _tausq_store))
  names!(res, vcat(:chain, :iter, :m1, :m0, :md, 
    [Symbol("beta" * "[$i]") for i in 0:p],
    [Symbol("lambdasq" * "[$i]") for i in 1:p],
    :tausq
  ))#, [Symbol("rd" * "$i") for i in 1:nints]))
  return res[(burnin+1):iter,:]
end

#gibbs(y,X,iter,burnin) = gibbs(y,X,iter,burnin,MersenneTwister())
;