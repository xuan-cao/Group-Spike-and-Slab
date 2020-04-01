library(mvtnorm)
library(truncnorm)
### X without intercept, beta0, gamma0 no intercept too.
### group contains the factors for each group, e.g. (1,1,1,2,2,2,3,3,3,...)
skinnygroup <- function(X, Y, group, ngroup, pr, nburn=1000,niter=5000, a0=0.01){
  n=nrow(X)
  p=ncol(X)
  del=0.01
  stau=  max(a0*ngroup^{2+del} /n,1.0)
  nu = 7.3
  sigsquare = (pi^2)*(nu -2)/(3*nu)
  outgamma = rep(0,ngroup)
  
  grouplistvec = list()
  groupvecn = rep(0,ngroup)
  for (j in 1:ngroup) {
    grouplistvec[[j]] = as.vector(which(group == j))
    groupvecn[j] = length(which(group == j))
  }
  

  
  # initializing. phi is 1/s^2 in the paper.
  beta = rep(0,p)
  gamma = rep(0,ngroup)
  gamma[sample(1:ngroup,5)] = 1
  phi = rep(1,n)
  Z = rep(0,n)
  
  # start updating
  for(itr in 1:(nburn+niter))
  {
    if(itr %% 100 == 0) print(paste("finish iteration", itr))

    
    for(j in 1:n) {
      # updating Y
      temp = t(X[j,]) %*% beta
      if(Y[j] > 0) Z[j] = rtruncnorm(1, a=0, mean = temp, sd = sqrt(1/phi[j]))
      else Z[j] = rtruncnorm(1, b=0, mean = temp, sd = sqrt(1/phi[j]))
      
      # updating phi
      phi[j] = rgamma(n = 1,shape = (1 + nu) / 2, rate = (Z[j] - temp)^2/2+sigsquare*nu/2)
    }
    
    #print(phi)
    for (g in 1:ngroup) {
      # update gamma
      qg = pr[g]
      betag = beta[grouplistvec[[g]]]
      Xg = X[, grouplistvec[[g]]]
      beta_g = beta[-grouplistvec[[g]]]
      X_g = X[, -grouplistvec[[g]]]
      invSigmag = t(Xg)%*%diag(phi)%*%Xg + diag(groupvecn[g])/(stau)
      Sigmag = solve(invSigmag)
      #print(det(Sigmag))
      mug = Sigmag%*%t(Xg)%*%diag(phi)%*%(Z - X_g%*%beta_g)
      temp = 1 + (qg)/(1-qg)*(stau)^(-groupvecn[g]/2)*det(Sigmag)^(0.5)*exp(0.5*t(mug)%*%invSigmag%*%mug)
      temp = 1 - 1/temp
      gamma[g] = rbinom(n = 1, size = 1, prob = temp)
      
      
      # update beta
      if(gamma[g] == 0) beta[grouplistvec[[g]]] = rep(0,groupvecn[g])
      else
        beta[grouplistvec[[g]]] = rmvnorm(n = 1, mean = mug, sigma = Sigmag)
    }

    
    #print(sum(gamma))
    if(itr > nburn) outgamma = outgamma + gamma
    
  
  }
  return(outgamma = outgamma/niter)
}





