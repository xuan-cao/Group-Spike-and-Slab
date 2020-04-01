rm(list = ls())
library(MASS)
library(truncnorm)
a0=0.01
b0=1

n = 100
p = 200
ngroup = 50
group = rep(seq(1,ngroup),rep(p/ngroup,ngroup))


pt=p/ngroup
set.seed(0)
t = 4
pt = t*pt

b = runif(pt,0.5,1.5) #setting 1
b = rep(1.5,pt) #setting 2
b = runif(pt,1.5,3) #setting 3
b = rep(3,pt) #setting 4

print(b)

Bc = c(b, array(0, p-pt)) #Coeffecients
true_group = c(rep(1, t), rep(0, ngroup - t))

rho1=0
rho2=0
rho3=0  ### correlation among active, between active and inactive, inactive parts
covr1=(1- rho1)*diag(pt) +  array(rho1,c(pt,pt))
covr3=(1- rho3)*diag(p-pt) +  array(rho3,c(p-pt,p-pt))
covr2=array(rho2,c(pt,p-pt))
covr=rbind(cbind(covr1,covr2),cbind(t(covr2),covr3))
covE = eigen(covr)

set.seed(0)

covsq = covE$vectors %*% diag(sqrt(covE$values)) %*% t(covE$vectors) 
Xs = matrix(rnorm(n*p), nrow = n)
Xn = covsq%*% t(Xs)
X=t(Xn)


Sigma = matrix(0,p,p)
for(j in 1:(p-1)){
  for(k in (j+1):p)
    Sigma[j,k] = Sigma[k,j] = 0.5^abs(j - k)
}
diag(Sigma) = 1

Sigma <- diag(p)




X <- mvrnorm(n,rep(0, p),Sigma=Sigma)

Y = rlogis(n, location = X %*% Bc)
Y = ifelse(Y>0,1,0)           # Logistic response E in the paper


n_test = 50
X_test = mvrnorm(n_test,rep(0, p),Sigma=Sigma)
Y_test = rlogis(n_test, location = X_test %*% Bc)
Y_test = ifelse(Y_test>0,1,0) 




source("skinnygroup_func_spike.R")
source("skinnygroup_func_neuron.R")





library(cvAUC)

gibbssen <- 0
gibbsspec <- 0
gibbsmcc <- 0
gibbsmspe <- 0 


neuronsen <- 0
neuronspec <- 0
neuronmcc <- 0
neuronmspe <- 0


lassosen <- 0
lassospec <- 0
lassomcc <- 0
lassomspe <- 0


scadsen <- 0
scadspec <- 0
scadmcc <- 0
scadmspe <- 0


mcpsen <- 0
mcpspec <- 0
mcpmcc <- 0
mcpmspe <- 0


gelsen <- 0
gelspec <- 0
gelmcc <- 0
gelmspe <- 0

repe <- 10
for(i in 1:repe){
############gibbs
print("gibbs")
starttime = Sys.time()
modelselection_gibbs = skinnygroup(X, Y, group=group, ngroup=ngroup, pr=rep(1/ngroup,ngroup), nburn=1500,niter=1500,a0=0.01)
endtime = Sys.time()
print(endtime - starttime)


a <- Evaluation(true_group, 1*(modelselection_gibbs>0.5))
gibbssen <- gibbssen + a$Recall
gibbsspec <- gibbsspec + a$Specific
gibbsmcc <- gibbsmcc + a$MCC


nonzerogroup <- which(modelselection_gibbs > 0.5)
nonzerobetaid <- which(group %in% nonzerogroup)
traindata <- cbind(Y, X[,nonzerobetaid])

fittedmodel <- glm(Y ~.-1, data = as.data.frame(traindata), family = binomial)
fit_beta <- as.vector(fittedmodel$coeff)
Y_hat_gibbs <- exp(X_test[,nonzerobetaid] %*% fit_beta)/(1 + exp(X_test[,nonzerobetaid] %*% fit_beta))
MSPE_gibbs <- round(mean((Y_test - Y_hat_gibbs)^2), digits = 4)


gibbsmspe <- gibbsmspe + MSPE_gibbs


library(pROC)
roc_obj <- roc(Y_test, Y_hat_gibbs)
auc(roc_obj)


############neuron
print("neuron")
starttime = Sys.time()
modelselection_neuron = skinnygroup_neuron(X, Y, group=group, ngroup=ngroup, pr=rep(1/ngroup,ngroup), nburn=1500,niter=1500,a0=0.01)
endtime = Sys.time()
print(endtime - starttime)


a <- Evaluation(true_group, 1*(modelselection_neuron>0.5))
neuronsen <- neuronsen + a$Recall
neuronspec <- neuronspec + a$Specific
neuronmcc <- neuronmcc + a$MCC


nonzerogroup <- which(modelselection_neuron > 0.5)
nonzerobetaid <- which(group %in% nonzerogroup)
traindata <- cbind(Y, X[,nonzerobetaid])

fittedmodel <- glm(Y ~.-1, data = as.data.frame(traindata), family = binomial)
fit_beta <- as.vector(fittedmodel$coeff)
Y_hat_neuron <- exp(X_test[,nonzerobetaid] %*% fit_beta)/(1 + exp(X_test[,nonzerobetaid] %*% fit_beta))
MSPE_neuron <- round(mean((Y_test - Y_hat_neuron)^2), digits = 4)
print(MSPE_neuron)


neuronmspe <- neuronmspe + MSPE_neuron


library(pROC)
roc_obj <- roc(Y_test, Y_hat_neuron)
auc(roc_obj)







#######group lasso
print("group lasso")
library(grpreg)
fit <- grpreg(X, Y, group, penalty="grLasso")
grlasso <- as.vector(coef(fit, lambda=0.0364))
a <- Evaluation(true_group, 1*(grlasso != 0))

lassosen <- lassosen + a$Recall
lassospec <- lassospec + a$Specific
lassomcc <- lassomcc + a$MCC



cvfit <- cv.grpreg(X, Y, group)
summary(cvfit)
Y_hat_grlasso <- as.vector(predict(fit, X_test, type="response", lambda=0.0364))
MSPE_grlasso <- round(mean((Y_test - Y_hat_grlasso)^2), digits = 4)
print(MSPE_grlasso)
library(pROC)
roc_obj <- roc(Y_test, Y_hat_grlasso)
auc(roc_obj)
lassomspe <- lassomspe + MSPE_grlasso



#######group SCAD
print("group SCAD")
library(grpreg)
fit <- grpreg(X, Y, group, penalty="grSCAD")
grSCAD <- as.vector(coef(fit, lambda=0.0473))
a <- Evaluation(true_group, 1*(grSCAD != 0))

scadsen <- scadsen + a$Recall
scadspec <- scadspec + a$Specific
scadmcc <- scadmcc + a$MCC



cvfit <- cv.grpreg(X, Y, group)
summary(cvfit)
Y_hat_grSCAD <- as.vector(predict(fit, X_test, type="response", lambda=0.0473))
MSPE_SCAD <- round(mean((Y_test - Y_hat_grSCAD)^2), digits = 4)
print(MSPE_SCAD)
roc_obj <- roc(Y_test, Y_hat_grSCAD)
auc(roc_obj)

scadmspe <- scadmspe + MSPE_SCAD



#######group MCP
X <- scale(X)
print("group MCD")
library(grpreg)
fit <- grpreg(X, Y, group, penalty="grMCP")
grMCP <- as.vector(coef(fit, lambda=0.0319))
a <- Evaluation(true_group, 1*(grMCP != 0))

mcpsen <- mcpsen + a$Recall
mcpspec <- mcpspec + a$Specific
mcpmcc <- mcpmcc + a$MCC



cvfit <- cv.grpreg(X, Y, group)
summary(cvfit)
Y_hat_grMCP <- as.vector(predict(fit, X_test, type="response", lambda=0.0319))
MSPE_MCP <- round(mean((Y_test - Y_hat_grMCP)^2), digits = 4)
print(MSPE_MCP)
roc_obj <- roc(Y_test, Y_hat_grMCP)
mcpmspe <- mcpmspe + MSPE_MCP




#######group gel
print("gel")
library(grpreg)
fit <- grpreg(X, Y, group, penalty="gel")
gel <- as.vector(coef(fit, lambda=0.0395))
a <- Evaluation(true_group, 1*(gel != 0))

gelsen <- gelsen + a$Recall
gelspec <- gelspec + a$Specific
gelmcc <- gelmcc + a$MCC

cvfit <- cv.grpreg(X, Y, group)
summary(cvfit)
Y_hat_gel <- as.vector(predict(fit, X_test, type="response", lambda=0.0395))
MSPE_gel <- round(mean((Y_test - Y_hat_gel)^2), digits = 4)
print(MSPE_gel)
roc_obj <- roc(Y_test, Y_hat_gel)
auc(roc_obj)
gelmspe <- gelmspe + MSPE_gel
}

print(gibbssen/repe)
print(gibbsspec/repe)
print(gibbsmcc/repe)
print(gibbsmspe/repe)


print(neuronsen/repe)
print(neuronspec/repe)
print(neuronmcc/repe)
print(neuronmspe/repe)


print(lassosen/repe)
print(lassospec/repe)
print(lassomcc/repe)
print(lassomspe/repe)


print(scadsen/repe)
print(scadspec/repe)
print(scadmcc/repe)
print(scadmspe/repe)


print(mcpsen/repe)
print(mcpspec/repe)
print(mcpmcc/repe)
print(mcpmspe/repe)


print(gelsen/repe)
print(gelspec/repe)
print(gelmcc/repe)
print(gelmspe/repe)



