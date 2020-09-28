# not working for some reason
# the Bayesian lasso
# Gibbs sampler parameterization by Park https://pdfs.semanticscholar.org/7c04/05e3d80ff04824335275a50c5921feba10ba.pdf
# but includes explicit intercept estimation

# to do: add logistic model

function initialize!(_beta, binits, p)
  if isnan(binits[1])
    _beta .= randn(p)*2
  else
    _beta .= copy(binits)
  end
end

function inits(y,X,Xint,iter,_mu_eta0,_tau_eta0, _sigmaa0, binits)
  # constants
  (N,p) = size(X)
  Nf = Float64(N)
  Ni = size(Xint[1], 1)
  munc, muint = Array{Float64,1}(undef, Ni), Array{Float64,1}(undef, Ni)
  X = hcat(ones(N), X)
  _beta = Array{Float64,1}(undef, p+1)
  initialize!(_beta, binits, p+1)
  mu = X * _beta
  res = y .- mu
  #Xint = Xint
  Xt = transpose(X)
  xtx = Xt * X
  xty = Xt * y
  _A::Symmetric{Float64, Array{Float64, 2}} = Symmetric(Array{Float64}(undef, p+1, p+1))
  _A_ms = Array{Float64,1}(undef, p+1)
  _iLams = Diagonal(vcat(_tau_eta0^(-2), fill(.01, p))) # precision prior on beta coefficients
  _muvec = vcat(_mu_eta0, zeros(Float64, p))      # mean prior on beta coefficients
  a = _sigmaa0 + Nf/2.
  _beta_store = Array{Float64,2}(undef, iter, p+1)
  _py_store = zeros(iter, 3)
  _lambda_store = zeros(iter)
  _sig_store = zeros(iter)
  _tau_store = Array{Float64, 2}(undef, iter, p)
  return(
    N, Nf, p, _beta, mu, 
    X, Xint, Xt, xtx, xty,
    _A, _A_ms, _iLams, _muvec,
    a, res,
    munc, muint,
    _beta_store, _py_store,_lambda_store,
    _sig_store, _tau_store
  )
end
inits(y,X,Xint,iter,_mu_eta0,_tau_eta0, _sigmaa0) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0, _sigmaa0, NaN)
inits(y,X,Xint,iter) = inits(y,X,Xint,iter,0,Inf, 0.0, NaN)

function updatebeta!(rng, _beta, _invsigma2, _iLams, _muvec, _A, _A_ms, y, xtx, xty)
  # block sampling of all model coefficients
  _A.data .= xtx .* _invsigma2 + _iLams 
  _A_ms .= xty .* _invsigma2 + _iLams * _muvec
  _beta .= rand(rng, MvNormalCanon(_A_ms,_A)) # parameters mu*Sigma^-1, Sigma^-1
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

function updatetausq!(rng, _lambdasq, _invsigma2, _invtausq, _iLams, _beta)
  lams2 =   _beta[2:end] .^ (-2.0) .* _lambdasq ./ _invsigma2
  # prior one from Park and Casella
  _invtausq[2:end] .= rand.(rng, InverseGaussian.(sqrt.(lams2), _lambdasq)) 
  #_invtausq .*= _invsigma2 # prior two from Park and Casella
  #
  #_iLams.diag[2:end] .= _invtausq[2:end]               # note itausq is just a view of _iLams, so this is not needed
end

function samplelambda(_r, _delta, _invtausq, p)
    sh = p + _r
    rt = _delta .+ 0.5*sum(inv.(_invtausq[2:end]))
    rand(rng, Gamma(sh, inv(rt))) # Gamma parmeterized by scale, rather than rate
end

function calcmd!(munc, muint, ymean, yscale, Xint, _beta)
  mul!(munc, Xint[1], _beta[2:end])
  mul!(muint, Xint[2], _beta[2:end])
  cm1 = ymean + yscale*(_beta[1] + mean(munc))
  cm0 = ymean + yscale*(_beta[1] + mean(muint))
  md = cm1-cm0
  cm1, cm0, md
end

function gibbs_lassolm(y,X,Xint,iter,burnin,rng; binits=NaN, chain = 1,
               offset = 0.0,
               _mu_eta0 = 0., _tau_eta0 = 1000.,                              # prior mean, sd of intercept
               _sigmaa0 = 0., _sigmab0 = 100.0,                                # precision priors
                ymean = 0.0, # rescaling coefficients
                yscale = 1.0, # rescaling coefficients
                _r = 1.0, _delta = 1.0  # lasso penalty shape, rate hyperpriors (conjugate gamma)
)

   (
    N, Nf, p, _beta, mu, 
    X, Xint, Xt, xtx, xty,
    _A, _A_ms, _iLams, _muvec,
    a, res,
    munc, muint,
    _beta_store, _py_store,_lambda_store,
    _sig_store, _tau_store
  ) = inits(y,X,Xint,iter,_mu_eta0,_tau_eta0, _sigmaa0, binits)
  
  # hierarchy not yet implemented
  #j = size(pl, 1)
  #pls = vcat(1, (cumsum(pl) .+ 1)[1:(end-1)])
  #plidx = [pls[g]:cumsum(pl)[g] for g in 1:j]
  #
  # priors
  # initials
  _lambdasq = 1.0
  _invsigma2 = rand()*4
  _invtausq = _iLams.diag; # - creates a view!
  for i in 1:iter
    ############
    # sample beta (with normal mixture prior, where mixing distribution)
    ############
    updatebeta!(rng, _beta, _invsigma2, _iLams, _muvec, _A, _A_ms, y, xtx, xty)
    #
    ############
    # sample sigma from InverseGamma
    ############
    _invsigma2 = sampleinvsigma2(rng, a, _sigmab0, res, y, X, _beta) # inverse gamma
    #_invsigma2 = sampleinvsigma2(rng, _sigmab0, res, y, X, _beta, 1.0, _invsigma2, Nf) # half-cauchy
    ############
    # sample tau from InverseGaussian (mixing distribution)
    ############
    updatetausq!(rng, _lambdasq, _invsigma2, _invtausq, _iLams, _beta)
    #
    ############
    # lasso penalty
    ############
    #_lambdasq = 1.0
    _lambdasq = samplelambda(_r, _delta, _invtausq, p)
    #
    if i > burnin  
      ####################
      # update mean difference
      ####################
      cm1, cm0, md = calcmd!(munc, muint, ymean, yscale, Xint, _beta)
      # store
      _py_store[i,:] = vcat(cm1, cm0, md)
      _beta_store[i,:] = _beta
      _tau_store[i,:] =_invtausq[2:end]
      _sig_store[i] = _invsigma2
      _lambda_store[i] = _lambdasq
    end
  end
    _tau_store = sqrt.(inv.(_tau_store))
    _sig_store = sqrt.(inv.(_sig_store))
    _lambda_store = sqrt.(_lambda_store)
  # this should be new template
  rr = hcat([chain for i in 1:iter], [i for i in 1:iter], _py_store, _beta_store, _tau_store, _sig_store, _lambda_store)
  nms = vcat(:chain, :iter, :m1, :m0, :md, [Symbol("beta" * "[$i]") for i in 0:p],
                  [Symbol("tau" * "[$i]") for i in 1:p],
                  :sigma, :lambda)
  if !isapprox(yscale, 1.0)
    _beta_ustd = _beta_store .* yscale
    _beta_ustd[:,1] .+= ymean
    rr = hcat(rr, _beta_ustd)
    nms = vcat(nms, [Symbol("betau" * "[$i]") for i in 0:p])
  end
  df = convert(DataFrame, rr)
  rename!(df, nms)
  df[(burnin+1):iter,:]
end

gibbs_lassolm(y,X,Xint,iter,burnin;thin=1,chain=1,ymean = 0.0, yscale=1.0, _r = 1.0, _delta = 1.0 ) = gibbs_lassolm(y,X,Xint, pl, iter, burnin, MersenneTwister(convert(Int, rand([i for i in 1:1e6])));thin=thin,chain=chain, ymean = ymean, yscale=yscale, _r = _r, _delta = _delta )
gibbs_lassolm(y,X,Xint,iter;thin=1,chain=1,ymean = 0.0, yscale=1.0, _r = 1.0, _delta = 1.0 ) = gibbs_lassolm(y,X,Xint, pl, iter, 0;thin=thin,chain=chain, ymean = ymean, yscale=yscale, _r = _r, _delta = _delta )
;