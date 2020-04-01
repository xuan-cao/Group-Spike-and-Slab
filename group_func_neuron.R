library(mvtnorm)
library(truncnorm)
### X without intercept, beta0, gamma0 no intercept too.
### group contains the factors for each group, e.g. (1,1,1,2,2,2,3,3,3,...)
skinnygroup_neuron <- function(X, Y, group, ngroup, pr, nburn=1000,niter=5000, a0=0.01){
  n=nrow(X)
  p=ncol(X)
  del=0.01
  stau=  max(a0*ngroup^{2+del} /n,1.0)
  nu = 7.3
  sigsquare = (pi^2)*(nu -2)/(3*nu)
  alpha0 = qnorm(1-pr[1])
  outgamma = rep(0,ngroup)
  
  T_func <- function(x){
    return(1 - 1*(x < alpha0))
  }
  
  grouplistvec = list()
  groupvecn = rep(0,ngroup)
  for (j in 1:ngroup) {
    grouplistvec[[j]] = as.vector(which(group == j))
    groupvecn[j] = length(which(group == j))
  }
  

  
  # initializing. phi is 1/s^2 in the paper.
  w = rep(0,p)
  alpha = rep(2,ngroup)
  alpha[sample(1:ngroup,5)] = 3
  gamma = sapply(alpha, T_func)
  D_alpha = diag(rep(gamma,groupvecn))
  beta = as.vector(D_alpha%*%w)
  phi = rep(1,n)
  Z = rep(0,n)
  
  # start updating
  for(itr in 1:(nburn+niter))
  {
    if(itr %% 100 == 0) print(paste("finish iteration", itr))

    
    for(j in 1:n) {
      # updating Z (Y in the paper)
      temp = t(X[j,]) %*% beta
      if(Y[j] > 0) Z[j] = rtruncnorm(1, a=0, mean = temp, sd = sqrt(1/phi[j]))
      else Z[j] = rtruncnorm(1, b=0, mean = temp, sd = sqrt(1/phi[j]))
      
      # updating phi
      phi[j] = rgamma(n = 1,shape = (1 + nu) / 2, rate = (Z[j] - temp)^2/2+sigsquare*nu/2)
    }
    
    # updating w
    W_half = diag(sqrt(phi))
    Phi_w = W_half%*%X%*%D_alpha
    alpha_w = W_half%*%Z
    D_w = diag(p)*stau
    Sigma_w = solve(t(Phi_w)%*%Phi_w +diag(p)/stau)
    
    
    u = rnorm(p, mean = 0, sd = sqrt(stau))
    delta = rnorm(n)
    nu_w = Phi_w%*%u + delta
    theta = solve(Phi_w%*%D_w%*%t(Phi_w) + diag(n))%*%(alpha_w - nu_w)
    w = u + D_w%*%t(Phi_w)%*%theta
    

    beta = as.vector(D_alpha%*%w)
    residual = Z - X%*%beta
    # update alpha
    for (g in 1:ngroup) {
      Xg = X[, grouplistvec[[g]]]
      wg = w[grouplistvec[[g]]]
      r_g = residual + T_func(alpha[g])*Xg%*%wg
      kappa = (1-pr[g])*exp(-t(r_g)%*%diag(phi)%*%r_g/2)/((1-pr[g])*exp(-t(r_g)%*%diag(phi)%*%r_g/2) + pr[g]*exp(-t(r_g - Xg%*%wg)%*%diag(phi)%*%(r_g - Xg%*%wg)/2))
      if(runif(1) < kappa) alpha[g] = rtruncnorm(1, b=alpha0, mean = 0, sd = 1)
      else alpha[g] = rtruncnorm(1, a=alpha0, mean = 0, sd = 1)
      residual = r_g - T_func(alpha[g])*Xg%*%wg
    }
    
    gamma = sapply(alpha, T_func)
    D_alpha = diag(rep(gamma,groupvecn))
    beta = as.vector(D_alpha%*%w)
    if(itr > nburn) outgamma = outgamma + gamma
    
  
  }
  return(outgamma = outgamma/niter)
}



Evaluation <- function(beta1, beta2){
  true.index <- which(beta1==1)
  false.index <- which(beta1==0)
  positive.index <- which(beta2==1)
  negative.index <- which(beta2==0)
  
  TP <- length(intersect(true.index,positive.index))
  FP <- length(intersect(false.index,positive.index))
  FN <- length(intersect(true.index,negative.index))
  TN <- length(intersect(false.index,negative.index))
  
  
  Precision <- TP/(TP+FP)
  if((TP+FP)==0) Precision <- 1
  Recall <- TP/(TP+FN)
  if((TP+FN)==0) Recall <- 1
  Sensitivity <- Recall
  Specific <- TN/(TN+FP)
  if((TN+FP)==0) Specific <- 1
  MCC.denom <- sqrt(TP+FP)*sqrt(TP+FN)*sqrt(TN+FP)*sqrt(TN+FN)
  if(MCC.denom==0) MCC.denom <- 1
  MCC <- (TP*TN-FP*FN)/MCC.denom
  if((TN+FP)==0) MCC <- 1
  
  return(list(Precision=Precision,Recall=Recall,Sensitivity=Sensitivity,Specific=Specific,MCC=MCC,TP=TP,FP=FP,TN=TN,FN=FN))
}

