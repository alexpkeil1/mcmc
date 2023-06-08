#!/usr/bin/env julia
# a series of gibbs variable selection functions
using Distributions, LinearAlgebra, DataFrames
# todo: standardize names of functions
# todo: eliminate alpha parameter



function inputchecks(y,X,Xint,p, pl)
  if !any(isnan.(Xint[1]))
    for xx in Xint
      @assert size(xx,2)==size(X,2) "X and Xint have different numbers of columns"
    end
  end
  @assert size(y,1) == size(X,1) "y and X have different numbers of rows"
  @assert sum(pl) == p "sum(pl) should equal number of columns of X (without intercept)"
end

function inits(rng, pl,iter,binits,y, X, Xint, _tau_eta0, _mu_tau0, _mu_eta0, _mu_mu0, _sigmaa0, pa, pb)
  # indexes
  (N,p) = size(X)
  j = size(pl, 1)
  inputchecks(y,X,Xint,p, pl)
  #pls = vcat(1, (cumsum(pl) .+ 1)[1:(end-1)])
  #plidx = [pls[g]:cumsum(pl)[g] for g in 1:j]
  pls = vcat(2, (cumsum(pl) .+ 2)[1:(end-1)])
  plidx = [pls[g]:cumsum(pl)[g]+1 for g in 1:j]
  # parameters
  _pi = rand.(rng, Beta.(pa,pb))
  _invsigma2 = rand()
  if size(binits, 1)==1 && isnan(binits)
    _beta = randn(rng, p+1)*2
  else
    _beta::Array{Float64, 1} = copy(binits) # if drawn from coef(glmfit), this can actually change the coefficients without copy()
  end
  # todo check length of mu/tau
  # draw initial selection from prior
  _gamma = vcat(1.0, vcat([Array{Float64,1}(rand(pp) .< _pi[g]) for (g, pp) in enumerate(pl)]...))
  _bg = _gamma .* _beta
  # hyper parameters
  _A_ms = Array{Float64, 1}(undef, p+1) 
  _A::Symmetric{Float64, Array{Float64, 2}} = Symmetric(Array{Float64}(undef, p+1, p+1))
  _iLams = Diagonal(vcat( _tau_eta0[1]^(-2), [ _mu_tau0[1]^(-2) for j in 1:p]))
  _muvec = vcat(_mu_eta0[1], [_mu_mu0[1] for j in 1:p])
  # storage
  _beta_store = zeros(iter, p+1)
  _gamma_store = zeros(iter, p)
  _bg_store = zeros(iter, p)
  _pi_store = zeros(iter, j)
  _sigma_store = zeros(iter)
  _py_store = zeros(iter, 3)
  # pre-allocations
  Nu = _gamma .* _beta
  Nuj = _gamma .* _beta
  X = hcat(ones(N), X)
  Xs = copy(X)
  makex!(Xs,_gamma)
  Xst = transpose(Xs)
  xstx = Xst * Xs
  munc  =  Xint[1] * _bg[2:end]
  muint = Xint[2] * _bg[2:end]
  yr = copy(y)
  bres = Array{Float64, 1}(undef, p+1)
  res = Array{Float64, 1}(undef, N) 
  # conveniences
  Nf = Float64(N)
  a = _sigmaa0 + Nf/2.

  return (
    Nf, p, pl, pls, plidx, j,
    _pi, _invsigma2, _beta, _gamma, _bg, 
    _beta_store, _gamma_store, _bg_store, _pi_store, _sigma_store, _py_store,
    Nu, Nuj, X, Xs, Xst, xstx, munc, muint, _iLams, _muvec, res, bres,
    _A_ms, _A, a, yr
  )
end

function makex!(Xs,_gamma; shift=0)
  @inbounds for (i,g) in enumerate(_gamma)
    Xs[:,i+shift] = Xs[:,i+shift]*g
  end
end

function makex!(X, Xs,_gamma;shift=0)
  copyto!(Xs,X)
  makex!(Xs,_gamma,shift=shift)
end

function fixp!(pj)
  if pj > one(Float64)
    println("pj above 0.0 - check for possible errors if this is frequent")
    #pj = one(Float64)
  elseif pj < zero(Float64)
    println("pj below 0.0 - check for possible errors if this is frequent")
    #pj = zero(Float64)
  elseif isnan(pj)
    println("pj is NaN - check for possible errors if this is frequent")
    #pj = zero(Float64)
  end
  pj
end


function sampleinvsigma2(rng, a, _sigmab0, res, y, X, _bg)
  # sample precision from a gamma prior
  res .= y .- X * _bg 
  se = transpose(res) * res #  permutedims(y .- X * _beta) * (y .- X * _beta)
  #a = _sigmaa0 + Nf/2.
  b = _sigmab0 + se/2.
  #sigma2 = rand(rng, InverseGamma(a, b))
  _invsigma2 = rand(rng, Gamma(a, 1. / b))
end


function sampleinvsigma2(rng, _sigmab0, res, y, X, _bg, nu, _invsigma2, Nf)
  # sample precision from a half cauchy prior (via representation as a scale mixture of gammas)
  va = 0.5 + nu/2.0
  vb = nu * _invsigma2 + _sigmab0^(-2) # _sigmab0 = A
  V = rand(rng, InverseGamma( va, vb))
  res .= y .- X * _bg 
  se = transpose(res) * res #  permutedims(y .- X * _beta) * (y .- X * _beta)
  #
  sa = (Nf + nu)/2.0
  sb = nu/V + se/2.
  _invsigma2 = rand(rng, Gamma(sa, 1. / sb))
end

function sampleinvsigma2_or(rng, a, _sigmab0, res, y, X, _bg;K=3,_invsigma2=_invsigma2)
  # sample precision from a gamma prior
  res .= y .- X * _bg 
  se = transpose(res) * res #  permutedims(y .- X * _beta) * (y .- X * _beta)
  #a = _sigmaa0 + Nf/2.
  b = _sigmab0 + se/2.
  #sigma2 = rand(rng, InverseGamma(a, b))
  orcand = rand(rng, Gamma(a, 1. / b), K)
  sp = sortperm(vcat(_invsigma2, orcand))
  selidx = K - sp[1]  +2 
  _invsigma2 = orcand[sp[2:end] .== selidx][1]
end


function sampleinvsigma2_or(rng, _sigmab0, res, y, X, _bg, nu, _invsigma2, Nf;K=3)
  # sample precision from a half cauchy prior (via representation as a scale mixture of gammas)
  va = 0.5 + nu/2.0
  vb = nu * _invsigma2 + _sigmab0^(-2) # _sigmab0 = A
  V = rand(rng, InverseGamma( va, vb))
  res .= y .- X * _bg 
  se = transpose(res) * res #  permutedims(y .- X * _beta) * (y .- X * _beta)
  #
  sa = (Nf + nu)/2.0
  sb = nu/V + se/2.
  orcand = rand(rng, Gamma(sa, 1. / sb), K)
  sp = sortperm(vcat(_invsigma2, orcand))
  selidx = K - sp[1]  +2 
  _invsigma2 = orcand[sp[2:end] .== selidx][1]
end


function updategamma!(rng, _gamma, _pi, _beta, Nuj, halfinvsigma2, res, y, X, j, plidx, pl, pa, pb)
  # variable selection indicators
  # algorithm given by Kuo and Mallick.
  # contrasts weighted normal likelihood under including vs. excluding a single coefficient to derive
  #  a probability of inclusion (weights equal to prior inclusion probabilities)
  # odds of inclusion are converted to a probability, which is used to take a Bernoulli draw
  rp = randperm(j)         #random order
  #rp = [pp for pp in 1:j] # fixed order
  @inbounds for q in rp                   # loop over hierarchical groups in random order
    vp = sample(plidx[q], size(plidx[q]), replace=false)  # random order (always excludes intercept)
    #vp = [pp for pp in plidx[q]]                         # fixed order
    piq = _pi[q]
    lpiq = log(piq)
    lnpiq = log(1-piq)
    for k in vp                               # loop over variables within group in random order
      Nuj .= _gamma .* _beta
      Nuj[k] = _beta[k]                       # first added in beta
      res  .= y .- X * Nuj
      rr =  transpose(res) * res
      cj = lpiq + halfinvsigma2 * rr            # proportional to likelihood*prior with inclusion
      res .+=  _beta[k] .* X[:,k]             # zero out the kth index (add in because of residual)
      rr =  transpose(res) * res
      dj = lnpiq + halfinvsigma2 * rr           # proportional to likelihood*prior with exclusion
      pj =  1. /(1. + exp( dj - cj))       # C*(lik*prior inclusion)/[C*(lik*prior exclusion) + C*(lik*prior exclusion) ] = posterior probability of inclusion
      _gamma[k] = rand(rng, Bernoulli(pj))
    end
  end
end

function updatepi!(rng, _pi, _gamma, _beta, Nuj, halfinvsigma2, res, y, X, j, plidx, pl, pa, pb)
  # update model inclusion priors via beta-bernoulli conjugacy
  @inbounds for q in 1:j
    g1 = sum(_gamma[plidx[q]])
    g0 = pl[q] - g1
    _pi[q] = rand(rng, Beta(pa[q] + g1, pb[q] + g0))
  end
end


function updatepi_or!(rng, _pi, _gamma, _beta, Nuj, halfinvsigma2, res, y, X, j, plidx, pl, pa, pb;K=3)
  # update model inclusion priors via beta-bernoulli conjugacy
  @inbounds for q in 1:j
    g1 = sum(_gamma[plidx[q]])
    g0 = pl[q] - g1
    orcand = rand(rng, Beta(pa[q] + g1, pb[q] + g0), K)
    sp = sortperm(vcat(_pi[q], orcand))
    selidx = K - sp[1]  +2 
    _pi[q]= orcand[sp[2:end] .== selidx][1]
  end
end


function sampleeta(rng, _bg, _invsigma2, _mu_eta0, _tau_eta0, yr, y, X, Nf)
  # model intercept
  yr .=  y .- X * _bg 
  #V = 1/(Nf * _invsigma2 + _tau_eta0^(-2))
  #M = (sum(yr) * _invsigma2 + _mu_eta0 * _tau_eta0^(-2) )*V
  #rand(rng, Normal(M, sqrt(V)))
  iS = Nf * _invsigma2 + _tau_eta0^(-2)
  AMS = sum(yr) * _invsigma2 + _mu_eta0 * _tau_eta0^(-2)
  rand(rng, NormalCanon(AMS, iS)) # parameters mu*Sigma^-1, Sigma^-1
end

function updatebeta!(rng, _beta, _bg, _gamma, _eta, _invsigma2, _iLams, _muvec, _A, _A_ms, yr, y, xstx, Xst)
  # block sampling of all non-intercept model coefficients
  yr .= y .- _eta
  _A.data .= xstx .* _invsigma2 + _iLams 
  _A_ms .= Xst * yr .* _invsigma2 + _iLams * _muvec
  _beta .= rand(rng, MvNormalCanon(_A_ms,_A)) # parameters mu*Sigma^-1, Sigma^-1
  _bg .= _gamma .* _beta
end

function updatebeta!(rng, _beta, _bg, _gamma, _invsigma2, _iLams, _muvec, _A, _A_ms, y, xstx, Xst)
  # block sampling of all model coefficients
  _A.data .= xstx .* _invsigma2 + _iLams 
  _A_ms .= Xst * y .* _invsigma2 + _iLams * _muvec
  _beta .= rand(rng, MvNormalCanon(_A_ms,_A)) # parameters mu*Sigma^-1, Sigma^-1
  _bg .= _gamma .* _beta
end


function updatebres!(bres, _beta, _mu, plidx, j)
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    bres[plidx[q]] .= bl .- _mu[q]
  end
end

function updatev!(rng, _V, _tausq, _tau_tau0, nu, j)
  @inbounds for q in 1:j # loop over hierarchical groups
    _V[q] = rand(rng, InverseGamma((nu+1.)/2.0, _tau_tau0[q]^(-2.0) + nu / _tausq[q]))
  end
end

function updatemu!(rng, _mu, _muvec, _beta, j, pl, plidx, _tausq, _mu_tau0, _mu_mu0, _alpha)    
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    iV = pl[q] / _tausq[q] +  _mu_tau0[q] .^(-2)
    MoV = (sum(bl) / _tausq[q] .+ _mu_mu0[q] * _mu_tau0[q] .^(-2))
    _mu[q] = rand(rng, NormalCanon(MoV, iV))
    _muvec[plidx[q]] .= _mu[q]
  end
end

function updatemu_robust!(rng, _mu, _muvec, _beta, _tausq, j, pl, plidx, _mu_tau0, _mu_mu0, _alpha)   
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    #iV = _alpha^2.0 * pl[q]/_tausq[q] + _mu_tau0[q] .^(-2)   
    iV = _alpha[q]^2.0 * pl[q]/_tausq[q] + _mu_tau0[q] .^(-2)   
    #MoV = _alpha * sum(bl)/_tausq[q] + _mu_mu0[q] * _mu_tau0[q] .^(-2)
    MoV = _alpha[q] * sum(bl)/_tausq[q] + _mu_mu0[q] * _mu_tau0[q] .^(-2)
    _mu[q] = rand(rng, NormalCanon(MoV, iV))
    #_muvec[plidx[q]] .= _mu[q]*_alpha
    _muvec[plidx[q]] .= _mu[q]*_alpha[q]
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

function updatetausq_robust!(rng, _beta, _tausq, _V, bres, j, pl, plidx, nu, _mu, _alpha)
  # note this doesn't give tau but gives a scaled version, which is scaled by sqrt(alpha_2)
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    #bres[plidx[q]] .= bl .- _mu[q]*_alpha
    bres[plidx[q]] .= bl .- _mu[q]*_alpha[q]
    bse = bres[plidx[q]] .^ 2.0
    a_tau = (pl[q] + nu )/2.
    b_tau = sum(bse)/2.0 + nu/_V[q] # this is a standard inverse gamma prior, so limit occurs as _tau_tau0[q] -> 0
    _tausq[q] = rand(rng, InverseGamma(a_tau, b_tau))
  end
end

function updatetausq_robust_or!(rng, _beta, _tausq, _V, bres, j, pl, plidx, nu, _mu, _alpha; K=9)
  # note this doesn't give tau but gives a scaled version, which is scaled by sqrt(alpha_2)
  # including ordered over-relaxation
  @inbounds for q in 1:j # loop over hierarchical groups
    bl = _beta[plidx[q]]
    #bres[plidx[q]] .= bl .- _mu[q]*_alpha
    bres[plidx[q]] .= bl .- _mu[q]*_alpha[q]
    bse = bres[plidx[q]] .^ 2.0
    a_tau = (pl[q] + nu )/2.
    b_tau = sum(bse)/2.0 + nu/_V[q] # this is a standard inverse gamma prior, so limit occurs as _tau_tau0[q] -> 0
    orcand = rand(rng, InverseGamma(a_tau, b_tau), K)
    sp = sortperm(vcat(_tausq[q], orcand))
    selidx = K - sp[1]  +2 
    _tausq[q] = orcand[sp[2:end] .== selidx][1]
  end
end


function samplealpha(rng, _beta, _mu, _tausq, pl, plidx, j; fix = true)
   if fix
     _alpha = 1.0
   else
     M,V = 0.0,0.0
     @inbounds for q in 1:j # loop over hierarchical groups
       M += _mu[q]*sum(_beta[plidx[q]])/_tausq[q]
       V += pl[q]*_mu[q]^2.0 /_tausq[q]
     end
     _alpha = rand(rng, NormalCanon(M,V)) # putting a prior on this because it can go to zero
   end
   _alpha
end

function updatealpha!(rng, _alpha, _beta, _mu, _tausq, pl, plidx, j; fix = true)
   # this is unidentified, but yields correct mu/beta (tau way off) but increases autocorrelation
   if fix
     _alpha .= ones(Float64, j)
   else
     @inbounds for q in 1:j # loop over hierarchical groups
       M = _mu[q]*sum(_beta[plidx[q]])/_tausq[q]
       V = pl[q]*_mu[q]^2/_tausq[q]
       _alpha[q] = rand(rng, NormalCanon(M,V)) # putting a prior on this because it can go to zero
     end
   end
end

function update_iLams!(rng, _iLams, _tausq, _alpha, plidx, j)
   #this is the appropriate covariance entry
   @inbounds for q in 1:j # loop over hierarchical groups
     _iLams.diag[plidx[q]] .= inv.(_tausq[q])   
   end
end

function updategprior!(_iLams, xtx, _invsigma2, G)
  _iLams[2:end,2:end] .= xtx[2:end,2:end] .* _invsigma2 / G
end

function calcmd!(munc, muint, ymean, yscale, Xint, _bg)
  mul!(munc, Xint[1], _bg[2:end])
  mul!(muint, Xint[2], _bg[2:end])
  cm1 = ymean + yscale*(_bg[1] + mean(munc))
  cm0 = ymean + yscale*(_bg[1] + mean(muint))
  md = cm1-cm0
  cm1, cm0, md
end

# gibbs variable selection (Kuo and Mallick)
function gvs_lm(y, X, Xint, pl::Array{Int64,1}, iter::Int64, burnin::Int64, rng; thin=1, chain=1,
               _mu_eta0 = 0., _tau_eta0 = 10.,                              # prior mean, sd of intercept
               _mu_mu0 = 0., _mu_tau0 = 1.0,                                # prior mean, sd of mu_l (beta means) (assumed constant over all beta)
               _sigmaa0 = 0.0, _sigmab0 = 5.0,                              # limit of uniform for prior shape scale of sigma^2 (model error term variance)
               pa = [1.0, 1.0], pb=[1.0, 1.0],
               binits = NaN,
               ymean = 1.0, # rescaling coefficients
               yscale = 1.0 # rescaling coefficients
               )
  ####################
  # initialize variables
  ####################
  (
    Nf, p, pl, pls, plidx, j,
    _pi, _invsigma2, _beta, _gamma, _bg, 
    _beta_store, _gamma_store, _bg_store, _pi_store, _sigma_store, _py_store,
    Nu, Nuj, X, Xs, Xst, xstx, munc, muint, _iLams, _muvec, res, bres,
    _A_ms, _A, a, yr
  ) =
  inits(rng, pl,iter,binits,y, X, Xint, _tau_eta0, _mu_tau0, _mu_eta0, _mu_mu0, _sigmaa0, pa, pb)
  nu = 1.0
  ####################
  # update loop
  ####################
  @inbounds for i in 1:iter
    ####################
    # update sigma
    ####################
    #_invsigma2 = sampleinvsigma2(rng, a, _sigmab0, res, y, X, _bg)                 # inverse-gamma (possibly flat with _sigmaa0, _sigmab0 = 0,0)
    _invsigma2 = sampleinvsigma2(rng, _sigmab0, res, y, X, _bg, nu, _invsigma2, Nf) # half cauchy
    ####################
    # update gamma, pi for each group
    ####################
    halfinvsigma2 = -0.5 * _invsigma2
    updategamma!(rng, _gamma, _pi, _beta, Nuj, halfinvsigma2, res, y, X, j, plidx, pl, pa, pb)
    updatepi!(rng, _pi, _gamma, _beta, Nuj, halfinvsigma2, res, y, X, j, plidx, pl, pa, pb)
    ####################
    # update beta
    ####################
    makex!(X, Xs,_gamma;shift=0)
    Xst .= transpose(Xs)
    mul!(xstx, Xst, Xs)
    updatebeta!(rng, _beta, _bg, _gamma, _invsigma2, _iLams, _muvec, _A, _A_ms, y, xstx, Xst)
    if i > burnin  
      ####################
      # update mean difference
      ####################
      cm1, cm0, md = calcmd!(munc, muint, ymean, yscale, Xint, _bg)
      ####################
      # store sweep values
      ####################  
      _pi_store[i,:] = _pi
      _sigma_store[i] = sqrt(1. / _invsigma2)
      _beta_store[i,:] =  _beta
      _gamma_store[i,:] = _gamma[2:end]
      _bg_store[i,:] =    _bg[2:end]
      _py_store[i,:] = vcat(cm1, cm0, md)
    end
  end
  rr = hcat([chain for i in 1:iter], [i for i in 1:iter], _py_store, 
                                _beta_store, _gamma_store, _bg_store, _pi_store, 
                                _sigma_store)
  nms = vcat(
       :chain, :iter,
       :m1, :m0, :md,
     [Symbol("beta_$i") for i in 0:p],
     [Symbol("gamma_$i") for i in 1:p],
     [Symbol("bxg_$i") for i in 1:p],
     [Symbol("pi_$i") for i in 1:j],
     :sigma
     )
  if !isapprox(yscale, 1.0)
    _beta_ustd = hcat(_beta_store[:,1], _bg_store) .* yscale
    _beta_ustd[:,1] .+= ymean
    rr = hcat(rr, _beta_ustd)
    nms = vcat(nms, [Symbol("b*gu" * "[$i]") for i in 0:p])
  end
  df = DataFrame(rr, nms)
  #df = convert(DataFrame, rr)
  #rename!(df, nms)
  df[range(burnin+1, iter, step=thin),:]
end
gvs_lm(y,X,Xint,pl::Array{Int64,1}, iter::Int64, burnin::Int64;thin=1,chain=1) = gvs_lm(y,X,Xint,pl,  iter, burnin, MersenneTwister(convert(Int, rand([i for i in 1:1e6])));thin=thin,chain=chain)
gvs_lm(y,X,Xint,pl::Array{Int64,1}, iter::Int64;thin=1,chain=1)         = gvs_lm(y,X,Xint,pl, iter, 0;thin=thin,chain=chain)



# gibbs variable selection (Kuo and Mallick with hierarchical priors on model coefficeints)
function gvs_hlm(y, X, Xint, pl::Array{Int64,1}, iter::Int64, burnin::Int64, rng; thin=1, chain=1,
               _mu_eta0 = 0.0, _tau_eta0 = 100.0,                              # prior mean, sd of intercept
               _mu_mu0 = [0.0], _mu_tau0 = [25.0],                # prior mean, scale of mu_l (beta group means)
               _tau_mu0 = [0.0], _tau_tau0 = [25.0],          # prior center, scale of half-t/half-cauchy (_tau_mu0 is unused)
               _sigmaa0 = 0.0, _sigmab0 = 5.0,                              # limit of uniform for prior shape scale of sigma^2 (model error term variance)
               pa = [1.0], pb=[1.0],                              # prior beta(a,b) parameters for pi (group selection priors)
               binits = NaN,
               nu = 1.0,
               ymean = 1.0, # rescaling coefficients
               yscale = 1.0, # rescaling coefficients
               fix = true # attempt parameter expansion (does not seem to work well)
               )
  # block sampler
  ####################
  # initialize variables
  ####################
  (
    Nf, p, pl, pls, plidx, j,
    _pi, _invsigma2, _beta, _gamma, _bg, 
    _beta_store, _gamma_store, _bg_store, _pi_store, _sigma_store, _py_store,
    Nu, Nuj, X, Xs, Xst, xstx, munc, muint, _iLams, _muvec, res, bres,
    _A_ms, _A, a, yr
  ) =
  inits(rng, pl,iter,binits,y, X, Xint, _tau_eta0, _mu_tau0, _mu_eta0, _mu_mu0, _sigmaa0, pa, pb)
  # hierarchical specific
  _alpha = rand(rng)*2
  _v_store = zeros(iter, j)
  _mu_store = zeros(iter, j)
  _tau_store = zeros(iter, j)
  _alpha_store = zeros(iter)
  _mu = randn(rng,j)*2
  _tausq = rand(rng,j)*2
  _V = 1.0 ./ _tausq
  ####################
  # update loop
  ####################
  @inbounds for i in 1:iter
    ####################
    # update sigma
    ####################
    #_invsigma2 = sampleinvsigma2(rng, a, _sigmab0, res, y, X, _bg)                 # inverse-gamma (possibly flat with _sigmaa0, _sigmab0 = 0,0)
    #_invsigma2 = sampleinvsigma2(rng, _sigmab0, res, y, X, _bg, nu, _invsigma2, Nf) # half cauchy
    _invsigma2 = sampleinvsigma2_or(rng, _sigmab0, res, y, X, _bg, nu, _invsigma2, Nf; K=21) # half cauchy
    ####################
    # update gamma, pi for each group
    ####################
    halfinvsigma2 = -0.5 * _invsigma2
    updategamma!(rng, _gamma, _pi, _beta, Nuj, halfinvsigma2, res, y, X, j, plidx, pl, pa, pb)
    #updatepi!(rng, _pi, _gamma, _beta, Nuj, halfinvsigma2, res, y, X, j, plidx, pl, pa, pb)
    updatepi_or!(rng, _pi, _gamma, _beta, Nuj, halfinvsigma2, res, y, X, j, plidx, pl, pa, pb; K=21)
    ####################
    # update tau and mu from t/Cauchy distribution
    ####################
    updatebres!(bres, _beta, _mu, plidx, j)
    updatev!(rng, _V, _tausq, _tau_tau0, nu, j)
    updatemu_robust!(rng, _mu, _muvec, _beta, _tausq, j, pl, plidx, _mu_tau0, _mu_mu0, _alpha)
    #updatetausq_robust!(rng, _beta, _tausq, _V, bres, j, pl, plidx, nu, _mu, _alpha)
    updatetausq_robust_or!(rng, _beta, _tausq, _V, bres, j, pl, plidx, nu, _mu, _alpha; K=21)    
    update_iLams!(rng, _iLams, _tausq, _alpha, plidx, j)
    
    _alpha = samplealpha(rng, _beta, _mu, _tausq, pl, plidx, j; fix = fix)
    #updatealpha!(rng, _alpha, _beta, _mu, _tausq, pl, plidx, j; fix = fix)

    ####################
    # update beta
    ####################
    # expand Lam, mu to pXp matrix
    makex!(X, Xs,_gamma;shift=0)
    Xst .= transpose(Xs)
    mul!(xstx, Xst, Xs)
    updatebeta!(rng, _beta, _bg, _gamma, _invsigma2, _iLams, _muvec, _A, _A_ms, y, xstx, Xst)

    if i > burnin  
      ####################
      # update mean difference
      ####################
      cm1, cm0, md = calcmd!(munc, muint, ymean, yscale, Xint, _bg)
      ####################
      # store sweep values
      ####################  
      _v_store[i,:] = _V
      _pi_store[i,:] = _pi
      _alpha_store[i] = _alpha
      _mu_store[i,:] = _mu .* _alpha
      _tau_store[i,:] = sqrt.(_tausq)
      _sigma_store[i] = sqrt(1. / _invsigma2)
      _beta_store[i,:] =  _beta
      _gamma_store[i,:] = _gamma[2:end]
      _bg_store[i,:] =    _bg[2:end]
      _py_store[i,:] = vcat(cm1, cm0, md)
    end
  end
  rr= hcat([chain for i in 1:iter], [i for i in 1:iter], _py_store, 
                                _beta_store, _gamma_store, _bg_store, _pi_store, 
                                _mu_store, _tau_store, _v_store,_alpha_store, _sigma_store)
  nms = vcat(
       :chain, :iter,
       :m1, :m0, :md,
     [Symbol("beta_$i") for i in 0:p],
     [Symbol("gamma_$i") for i in 1:p],
     [Symbol("bxg_$i") for i in 1:p],
     [Symbol("pi_$i") for i in 1:j],
     [Symbol("mu" * "[$i]") for i in 1:j],
     [Symbol("tau" * "[$i]") for i in 1:j],
     [Symbol("V" * "[$i]") for i in 1:j],
     [Symbol("alpha" * "[$i]") for i in 1:1],
     :sigma
     )                              
  if !isapprox(yscale, 1.0)
    _beta_ustd = hcat(_beta_store[:,1], _bg_store) .* yscale
    _beta_ustd[:,1] .+= ymean
   rr = hcat(rr, _beta_ustd)
    nms = vcat(nms, [Symbol("b*gu" * "[$i]") for i in 0:p])
  end
  df = DataFrame(rr, nms)
  #df = convert(DataFrame, rr)
  #rename!(df, nms)
  df[range(burnin+1, iter, step=thin),:]

end
gvs_hlm(y,X,Xint,pl::Array{Int64,1}, iter::Int64, burnin::Int64;thin=1,chain=1,nu=1.0, fix = true) = gvs_hlm(y,X,Xint,pl,  iter, burnin, MersenneTwister(convert(Int, rand([i for i in 1:1e6])));thin=thin,chain=chain, nu=nu,fix = fix)
gvs_hlm(y,X,Xint,pl::Array{Int64,1}, iter::Int64;thin=1,chain=1,nu=1.0, fix = true)         = gvs_hlm(y,X,Xint,pl, iter, 0;thin=thin,chain=chain, nu=nu,fix = fix)


# gibbs variable selection with Zellner's g-prior
# this suffers from getting "stuck" when pi is close to zero for one group
# could benefit from parameter expansion?
function gvs_zlm(y, X, Xint, pl::Array{Int64,1}, iter::Int64, burnin::Int64, rng; thin=1, chain=1,
               _mu_eta0 = 0., _tau_eta0 = 10.,                              # prior mean, sd of intercept
               _mu_mu0 = 0., _mu_tau0 = 1.0,                            # not actually used here
               _sigmaa0 = 0.0, _sigmab0 = 5.0,                              # limit of uniform for prior shape scale of sigma^2 (model error term variance)
               pa = [1.0, 1.0], pb=[1.0, 1.0],
               G=6889.,                                                     # the g prior
               binits = NaN,
               ymean = 1.0, # rescaling coefficients
               yscale = 1.0 # rescaling coefficients
               )
  ####################
  # initialize variables
  ####################
  (
    Nf, p, pl, pls, plidx, j,
    _pi, _invsigma2, _beta, _gamma, _bg, 
    _beta_store, _gamma_store, _bg_store, _pi_store, _sigma_store, _py_store,
    Nu, Nuj, X, Xs, Xst, xstx, munc, muint, _iLams, _muvec, res, bres,
    _A_ms, _A, a, yr
  ) =
  inits(rng, pl,iter,binits,y, X, Xint, _tau_eta0, _mu_tau0, _mu_eta0, _mu_mu0, _sigmaa0, pa, pb)
  # zellner specific 
  _iLams = diagm(vcat( _tau_eta0[1]^(-2), [ _mu_tau0[1]^(-2) for j in 1:p])) # this gets overwritten except row/col 1
  xtx = transpose(X)*X
  nu = 1.0
  ####################
  # update loop
  ####################
  @inbounds for i in 1:iter
    ####################
    # update sigma
    ####################
    #_invsigma2 = sampleinvsigma2(rng, a, _sigmab0, res, y, X, _bg)                 # inverse-gamma (possibly flat with _sigmaa0, _sigmab0 = 0,0)
    _invsigma2 = sampleinvsigma2(rng, _sigmab0, res, y, X, _bg, nu, _invsigma2, Nf) # half cauchy
    ####################
    # update gamma, pi for each group
    ####################
    halfinvsigma2 = -0.5 * _invsigma2
    updategamma!(rng, _gamma, _pi, _beta, Nuj, halfinvsigma2, res, y, X, j, plidx, pl, pa, pb)
    updatepi!(rng, _pi, _gamma, _beta, Nuj, halfinvsigma2, res, y, X, j, plidx, pl, pa, pb)
    ####################
    # update g prior
    ####################
    updategprior!(_iLams, xtx, _invsigma2, G)
    ####################
    # update beta
    ####################
    makex!(X, Xs,_gamma;shift=0)
    Xst .= transpose(Xs)
    mul!(xstx, Xst, Xs)
    updatebeta!(rng, _beta, _bg, _gamma, _invsigma2, _iLams, _muvec, _A, _A_ms, y, xstx, Xst)
    if i > burnin  
      ####################
      # update mean difference
      ####################
      cm1, cm0, md = calcmd!(munc, muint, ymean, yscale, Xint, _bg)
      ####################
      # store sweep values
      ####################  
      _pi_store[i,:] = _pi
      _sigma_store[i] = sqrt(1. / _invsigma2)
      _beta_store[i,:] =  _beta
      _gamma_store[i,:] = _gamma[2:end]
      _bg_store[i,:] =    _bg[2:end]
      _py_store[i,:] = vcat(cm1, cm0, md)
    end
  end
  rr = hcat([chain for i in 1:iter], [i for i in 1:iter], _py_store, 
                                _beta_store, _gamma_store, _bg_store, _pi_store, _sigma_store)
  nms = vcat(
       :chain, :iter, :m1, :m0, :md,
     [Symbol("beta_$i") for i in 0:p], [Symbol("gamma_$i") for i in 1:p],
     [Symbol("bxg_$i") for i in 1:p], [Symbol("pi_$i") for i in 1:j],
     :sigma
     )                              
  if !isapprox(yscale, 1.0)
    _beta_ustd = hcat(_beta_store[:,1], _bg_store) .* yscale
    _beta_ustd[:,1] .+= ymean
    rr = hcat(rr, _beta_ustd)
    nms = vcat(nms, [Symbol("b*gu" * "[$i]") for i in 0:p])
  end
  df = DataFrame(rr, nms)
  #df = convert(DataFrame, rr)
  #rename!(df, nms)
  df[range(burnin+1, iter, step=thin),:]
end
gvs_zlm(y,X,Xint,pl::Array{Int64,1}, iter::Int64, burnin::Int64;thin=1,chain=1,G=6889.) = gvs_zlm(y,X,Xint,pl,  iter, burnin, MersenneTwister(convert(Int, rand([i for i in 1:1e6])));thin=thin,chain=chain,G=G)
gvs_zlm(y,X,Xint,pl::Array{Int64,1}, iter::Int64;thin=1,chain=1,G=6889.)         = gvs_zlm(y,X,Xint,pl, iter, 0;thin=thin,chain=chain,G=G)

;