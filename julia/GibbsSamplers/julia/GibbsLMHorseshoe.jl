# still in progress - need more function calls to eliminate Xint, generalize to different scales
using Distributions, DataFrames, GLM, StatsBase, LinearAlgebra

# to do: add logistic model



# linear model
#function gibbs_horseshoelm(y,X,Xint,pl, iter, burnin, rng; 
#  offset = 0.0, binits=NaN, # not yet implemented
#  thin=1, chain=1, 
#  ymean = 0.0, # rescaling coefficients
#  yscale = 1.0 # rescaling coefficients
#)
#  # block sampler
#  (N::Int64,p::Int64) = size(X)
#  numgroups::Int64 = size(pl, 1)
#  if sum(pl) != p
#    throw("sum(pl) should equal number of columns of X (without intercept)")
#  end
#  j = size(pl, 1)
#  #pls = vcat(1, (cumsum(pl) .+ 1)[1:(end-1)])
#  #plidx = [pls[g]:cumsum(pl)[g] for g in 1:j]
#  pls = vcat(2, (cumsum(pl) .+ 2)[1:(end-1)])
#  plidx = [pls[g]:cumsum(pl)[g]+1 for g in 1:j]
#
#  # constants/hyperpriors
#  # priors
#  _mu_eta0::Float64, _mu_eta1::Float64 = 0., 100000.         # intercept priors
#  _mu_etaZ::Float64 = _mu_eta0/_mu_eta1^(-2)
#  _sigma0::Float64, _sigma1::Float64 = 0.0001, 0.0001          # prior a,b parameters for sigma (model error)
#  # data transformations
#  Xt::Array{Float64,2} = permutedims(X)
#  xtx::Array{Float64,2} = Xt * X
#  _kappa::Array{Float64,1} = y .- 0.5
#  # initialize storage vectors
#  _py_store::Array{Float64,2}  = Array{Float64}(undef, iter, 3)
#  _sigma_store::Array{Float64,1}  = Array{Float64}(undef, iter)
#  _beta_store::Array{Float64,2}  = Array{Float64}(undef, iter, p+1)
#  _lamsq_store::Array{Float64,2}  = Array{Float64}(undef, iter, p)
#  _tausq_store::Array{Float64,2}  = Array{Float64}(undef, iter, numgroups)
#  # alt containers
#  #_A::Array{Float64,2}= Array{Float64}(undef, p, p)
#  _invLams::Diagonal{Float64,Array{Float64,1}} = Diagonal(Array{Float64}(undef, p, p))
#  # uninitialized
#  _beta::Array{Float64,1} = rand(p+1)
#   _nu::Array{Float64, 1} = Array{Float64}(undef, p)
#  _zeta::Array{Float64,1} = Array{Float64}(undef, numgroups)
#  _A::Symmetric{Float64, Array{Float64, 2}} = Symmetric(Array{Float64}(undef, p, p))
#  _As::Array{Float64, 2} = Array{Float64}(undef, p, p)
#  _A_ms::Array{Float64, 1} = Array{Float64}(undef, p)
#  # inits
#  _sigsq::Float64 = rand()*2
#  _eta::Float64 = randn()*2
#  _tausq::Array{Float64,1} = rand(numgroups)*2
#  _lamsq::Array{Float64,1} =  rand(p) .* 2
#   muint::Array{Float64,1}  = Array{Float64}(undef, N)
#   munc::Array{Float64,1}  = Array{Float64}(undef, N)
#
#  for g in 1:numgroups, j in plidx[g]
#      _invLams[j,j] = 1 ./ (_tausq[g] *  _lamsq[j])
#  end
#  # Utility
#  siga::Float64 = N/2.
#  nones::Array{Float64, 1} = ones(Float64, N)
#  yr = copy(y)
#  xb = copy(y)
#  @inbounds for i in 1:iter
#    xb .=  X * _beta
#    ####################
#    # update sigma
#    ####################
#    sse = 0.0
#    @inbounds for j in 1:N
#      sse += (y[j] - _eta - xb[j])^2.
#    end
#    sse += (permutedims(_beta) * _invLams * _beta)[1] # sse variable recycled for efficiency
#    sse /= 2.
#    _sigsq = rand(rng, InverseGamma(siga + p/2., sse))
#    ####################
#    # update shrinkage parameters
#    ####################
#      # global shrinkage
#      @inbounds for g in 1:numgroups 
#        _zeta[g] = rand(rng, InverseGamma(
#                1, 
#                1 + 1/_tausq[g]
#               ))
#        _tausq[g] = rand(rng, InverseGamma(
#                ( pl[g]+1 )/2, 
#                1/_zeta[g] + 1 / ( 2*_sigsq )*sum(_beta[plidx[g]] .^2 ./ _lamsq[plidx[g]] )
#               )) 
#        # local shrinkage
#        for j in plidx[g]
#          _nu[j] = rand(rng, InverseGamma(
#                    1,
#                    1 .+ 1 ./ _lamsq[j]
#                    ))
#          _lamsq[j] = rand(rng, InverseGamma(
#                    1,
#                    1 / _nu[j] + _beta[j]^2 ./ (2 * _tausq[g] * _sigsq )
#                    ))
#          _invLams[j,j] = 1. / (_tausq[g] * _lamsq[j])
#        end
#      end
#      _A.data .= Symmetric(xtx + _invLams)
#    ####################
#    # update eta
#    ####################
#    xb .=  X * _beta
#    yr .=  y .- xb
#    # Efficient and avoids inversion of iV
#    iV = N / _sigsq + 1 / (_mu_eta1^2)
#    MiV = reduce(+, yr) / _sigsq + _mu_etaZ 
#    _eta = rand(rng, NormalCanon(MiV,iV))
#    ####################
#    # update beta
#    ####################
#    yr .= y .- _beta[1]
#    # Efficient and avoids inversion of _A
#    _A_ms .= Xt * yr / _sigsq
#    _As .= _A ./ _sigsq
#    _beta[2:end] .= rand(rng, MvNormalCanon(_A_ms,_As)) # parameters mu*Sigma^-1, Sigma^-1
#    #
#    ####################
#    # update mean difference
#    ####################
#    cm1, cm0, md = calcmd!(munc, muint, ymean, yscale, Xint, _beta)
#    ####################
#    # store sweep values
#    ####################    
#    _sigma_store[i] = sqrt(_sigsq)
#    _beta_store[i,:] = vcat(_eta, _beta)
#    _lamsq_store[i,:] = _lamsq
#    _tausq_store[i,:] = _tausq
#    _py_store[i,:] = vcat(cm1, cm0, md)
#  end
#  df = convert(DataFrame, hcat([chain for i in 1:iter], [i for i in 1:iter], _py_store, _beta_store, _lamsq_store, _tausq_store, _sigma_store))
#  rename!(df, vcat(
#       :chain, :iter,
#       :m1, :m0, :md,
#     [Symbol("beta" * "[$i]") for i in 0:(p)],
#     [Symbol("lambdasq" * "[$i]") for i in 1:(p)],
#     [Symbol("tausq" * "[$i]") for i in 1:(numgroups)],
#     :sigma
#     ))
#  df[range(burnin+1, iter, step=thin),:]
#end
#gibbs_horseshoelm(y,X,Xint, pl, iter, burnin;offset = 0.0, binits=NaN,thin=1,chain=1,ymean = 0.0, yscale=1.0) = gibbs_horseshoelm(y,X,Xint, pl, iter, burnin, MersenneTwister(convert(Int, rand([i for i in 1:1e6])));offset = offset, binits=binits, thin=thin,chain=chain, ymean = ymean, yscale=yscale)
#gibbs_horseshoelm(y,X,Xint, pl, iter;offset = 0.0, binits=NaN,thin=1,chain=1,ymean = 0.0, yscale=1.0) = gibbs_horseshoelm(y,X,Xint, pl, iter, 0;offset = offset, binits=binits,thin=thin,chain=chain, ymean = ymean, yscale=yscale)



function initialize!(_beta, binits, p)
  if isnan(binits[1])
    _beta .= randn(p)*2
  else
    _beta .= copy(binits)
  end
end

function inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, _sigmaa0,binits)
  # constants
  (N,p) = size(X)
  Nf = Float64(N)
  Ni = size(Xint[1], 1)
  a = _sigmaa0 + Nf/2.
  X = hcat(ones(N), X)
  _beta = Array{Float64,1}(undef, p+1)
  initialize!(_beta, binits, p+1)
  mu = X * _beta
  Xint = Xint
  Xt = transpose(X)
  xtx = Xt * X
  xty = Xt * y
  _A::Symmetric{Float64, Array{Float64, 2}} = Symmetric(Array{Float64}(undef, p+1, p+1))
  _A_ms = Array{Float64,1}(undef, p+1)
  _iLams = vcat(_tau_eta0^(-2), [_tau_beta0^(-2) for j in 1:p]) # precision prior on beta coefficients
  _muvec = vcat(_mu_eta0, [_mu_beta0 for j in 1:p])      # mean prior on beta coefficients
  munc, muint = Array{Float64,1}(undef, Ni), Array{Float64,1}(undef, Ni)
  #z = kappa ./ om
  _beta_store = Array{Float64,2}(undef, iter, p+1)
  _py_store = zeros(iter, 3)
  _sig_store= Array{Float64,1}(undef, iter)
  return(
    N, Nf, p, _beta, mu,
    X, Xint, Xt, xtx, xty,
    _A, _A_ms, _iLams, _muvec, a, 
    munc, muint,
    _beta_store, _py_store, _sig_store
  )
end

inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, NaN)
inits(y,X,Xint,iter) = inits(y,X,Xint,iter,0.0,Inf,0.0,Inf, NaN)

function updatenu!(_nu, _lamsq, p)
  # local shrinkage latent variable (p length)
  na = 1.0
  nb = 1.0 .+ inv.(_lamsq)
  _nu .= rand.(rng, InverseGamma.(na,nb))
end

function updatelamsq!(_lamsq, _tausq, _invsigma2, _nu, _beta, p)
  # local shrinkage parameter (p length)
  la = 1.0
  lb = inv.(_nu) 
  lb .+= _beta .^2.0 .* _invsigma2 ./ (2.0 .* _tausq )
  _lamsq .= rand.(rng, InverseGamma.(la,lb))
end

function samplezeta(_tausq)
  # global shrinkage latent variable
  za = 1.0
  zb = 1.0 + inv(_tausq)
  rand(rng, InverseGamma( za, zb ))
end

function sampletausq(_invsigma2, _lamsq, _zeta, _beta, p)
  # global shrinkage parameter
  ta = ( p + 1.0 ) / 2.0
  tb = inv(_zeta) 
  tb += 0.5 * _invsigma2  * sum(_beta .^2.0 ./ _lamsq )
  rand(rng, InverseGamma(ta, tb))
end

function updateiLams!(_iLams, _tausq, _lamsq)
  _iLams[2:end] .= inv.(_tausq .* _lamsq)
end

function sampleinvsigma2(rng, a, _sigmab0, res, y, X, _beta)
  # sample precision from a gamma prior
  res .= y .- X * _beta 
  se = transpose(res) * res #  permutedims(y .- X * _beta) * (y .- X * _beta)
  #a = _sigmaa0 + Nf/2.
  b = _sigmab0 + se/2.
  #sigma2 = rand(rng, InverseGamma(a, b))
  rand(rng, Gamma(a, 1. / b))
end


function sampleinvsigma2(rng, _sigmab0, res, y, X, _beta, nu, _invsigma2, Nf)
  # sample precision from a half cauchy prior (via representation as a scale mixture of gammas)
  va = 0.5 + nu/2.0
  vb = nu * _invsigma2 + _sigmab0^(-2) # _sigmab0 = A
  V = rand(rng, InverseGamma( va, vb))
  res .= y .- X * _beta 
  se = transpose(res) * res #  permutedims(y .- X * _beta) * (y .- X * _beta)
  #
  sa = (Nf + nu)/2.0
  sb = nu/V + se/2.
  rand(rng, Gamma(sa, 1. / sb))
end


function updatebeta!(rng, _beta, xtx, xty, _iLams, _muvec, _A, _A_ms)
   # block sampler for beta coefficients
   _A.data .= xtx + Diagonal(_iLams)
   _A_ms .= xty .+ _iLams .* _muvec   # for vector of precisions (iLams is diagonal of precision matrix)
  _beta .= rand(rng, MvNormalCanon(_A_ms, _A))
end

function calcmd!(munc, muint, ymean, yscale, Xint, _beta)
  mul!(munc, Xint[1], _beta[2:end])
  mul!(muint, Xint[2], _beta[2:end])
  cm1 = ymean + yscale*(_beta[1] + mean(munc))
  cm0 = ymean + yscale*(_beta[1] + mean(muint))
  md = cm1-cm0
  cm1, cm0, md
end

function gibbs_horseshoelm(y,X,Xint,iter,burnin,rng; chain=1,
               binits=NaN,
                ymean = 0.0, # rescaling coefficients
                yscale = 1.0, # rescaling coefficients
               offset = 0.0,
               _mu_eta0 = 0., _tau_eta0 = 1000.,                              # prior mean, sd of intercept
               _mu_beta0 = 0., _tau_beta0 = 100.0,                                # prior mean, sd (ignored) of beta
               _sigmaa0 = 0.0, _sigmab0 = 0.0
)
  (
    N, Nf, p, _beta, mu,
    X, Xint, Xt, xtx, xty,
    _A, _A_ms, _iLams, _muvec, a, 
    munc, muint,
    _beta_store, _py_store, _sig_store
  ) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0,_mu_beta0,_tau_beta0, _sigmaa0,binits)
  #
  res = zeros(N)
  _nu = rand(rng, p)
  _invsigma2 = rand(rng)
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
    updatelamsq!(_lamsq, _tausq, _invsigma2, _nu, _beta[2:end], p)
    ##############################
    # update local shrinkage
    ##############################
    _zeta = samplezeta(_tausq)
    _tausq = sampletausq(_invsigma2, _lamsq, _zeta, _beta[2:end], p)
    updateiLams!(_iLams, _tausq, _lamsq)
    ##############################
    # update model parameters
    ##############################
    updatebeta!(rng, _beta, xtx, xty, _iLams, _muvec, _A, _A_ms)
    ############
    # sample sigma from InverseGamma
    ############
    _invsigma2 = sampleinvsigma2(rng, a, _sigmab0, res, y, X, _beta) # inverse gamma
    #_invsigma2 = sampleinvsigma2(rng, _sigmab0, res, y, X, _beta, 1.0, _invsigma2, Nf) # half-cauchy
    if i > burnin  
      ##############################
      # update summary parameters
      ##############################
      cm1, cm0, md = calcmd!(munc, muint, ymean, yscale, Xint, _beta)
      ##############################
      # storage
      ##############################
      _sig_store[i] = _invsigma2
      _tausq_store[i] = _tausq
      _lamsq_store[i,:] = _lamsq
      _beta_store[i,:] = _beta
      _py_store[i,:] = vcat(cm1, cm0, md)
    end
  end
  # transform and clean output
  _sig_store = inv.(sqrt.(_sig_store))
  rr = hcat([chain for i in 1:iter], [i for i in 1:iter], _py_store, _beta_store, _lamsq_store, _tausq_store, _sig_store)
  nms = vcat(:chain, :iter, :m1, :m0, :md, 
    [Symbol("beta" * "[$i]") for i in 0:p],
    [Symbol("lambdasq" * "[$i]") for i in 1:p],
                  :tausq, :sigma)
  if !isapprox(yscale, 1.0)
    _beta_ustd = _beta_store .* yscale
    _beta_ustd[:,1] .+= ymean
    rr = hcat(rr, _beta_ustd)
    nms = vcat(nms, [Symbol("betau" * "[$i]") for i in 0:p])
  end
  df = DataFrame(rr, nms)
  #df = convert(DataFrame, rr)
  #rename!(df, nms)
  df[(burnin+1):iter,:]
end

gibbs_horseshoelm(y,X,Xint, iter, burnin;offset = 0.0, binits=NaN,thin=1,chain=1,ymean = 0.0, yscale=1.0) = gibbs_horseshoelm(y,X,Xint, iter, burnin, MersenneTwister(convert(Int, rand([i for i in 1:1e6])));offset = offset, binits=binits, thin=thin,chain=chain, ymean = ymean, yscale=yscale)
gibbs_horseshoelm(y,X,Xint, iter;offset = 0.0, binits=NaN,thin=1,chain=1,ymean = 0.0, yscale=1.0) = gibbs_horseshoelm(y,X,Xint, iter, 0;offset = offset, binits=binits,thin=thin,chain=chain, ymean = ymean, yscale=yscale)
;