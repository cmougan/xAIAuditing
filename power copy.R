############
# 
# Power analysis of testing Accuracy vs AUC
#
############

# cleaning environment
rm(list = ls())

# redo time-costly experiments or reload from saved workspace
redoexp = TRUE

# load packages (and install them, if needed)
llibrary = function(packages) {
  new.packages = packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new.packages)) install.packages(new.packages)
  for( p in packages) {
    require(p, character.only = TRUE)
  }  
}

# required packages
llibrary( c("mvtnorm", "caret", "pROC", "brunnermunzel", "future.apply") ) 

# set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) 

# if workspace was saved, reload
if(!redoexp)
  load("power.RData")

# random bivariate dataset
rbiv = function(n, mu=0, cov=0.5, p=0.5) {
  covmat0 = matrix(c(1, cov, cov, 1), nrow=2, byrow=TRUE)
  covmat1 = matrix(c(1, cov, cov, 1), nrow=2, byrow=TRUE)
  # C ~ Ber(p)
  n1 = rbinom(1, n, p)
  n0 = n - n1
  # P(X|C=0) ~ Bivariate Normal fbiv0
  xy0 = rmvnorm(n0, c(mu, mu), covmat0)
  df0 = data.frame(xy0)
  names(df0) = c("x", "y")
  df0$c = 0
  # P(X|C=1) ~ Bivariate Normal fbiv1
  xy1 = rmvnorm(n1, c(-mu, -mu), covmat1)
  df1 = data.frame(xy1)
  names(df1) = c("x", "y")
  df1$c = 1
  res = rbind(df0, df1)
  res$c = as.factor(res$c)
  res
}

classifierTest = function(df_train, df_test, alpha) {
  lr.fit = train(c~x+y, data=df_train, method="glm", family=binomial(logit))
  lr.pred = predict(lr.fit, newdata = df_test)
  lr.prob = predict(lr.fit, newdata = df_test, type="prob") 
  lr.score = lr.prob[,2] 
  min_accuracy = max(table(df_train$c)) / nrow(df_train)
  bt = binom.test(sum(lr.pred==df_test$c), length(lr.pred), min_accuracy, conf.level=1-alpha, alternative="greater")
  negpos = split(lr.score, df_test$c)
  bm = brunnermunzel.test(negpos$`0`, negpos$`1`, alpha=alpha, alternative="less")
  #
  n = length(negpos$`1`)
  m = length(negpos$`0`)
  nx = (n+m)/2
  v = 0.25/(n*m)*(2*nx-1+(3*nx-3)/2.25)
  T = (bm$estimate-0.5)/sqrt(v)
  wtp = 1-pnorm(T)
  unname(c(bt$p.value, bt$conf.int, bm$p.value, bm$conf.int, wtp))
}

gentest = function(alpha, mu, p) {
  df_train = rbiv(1400, mu, p=p)
  df_test = rbiv(600, mu, p=p)
  res = classifierTest(df_train, df_test, alpha=alpha)
  c(res[1], res[4], res[7])
}

runtest = function(alpha, mu, p) {
  res = future_replicate(1000, gentest(alpha, mu, p))
  c(mean(res[1,]<alpha),mean(res[2,]<alpha),mean(res[3,]<alpha))
}

runtest05 = function(mu) runtest(alpha=0.05, mu=mu, p=0.5)
runtest02 = function(mu) runtest(alpha=0.05, mu=mu, p=0.2)

plan(multisession, workers = 16)
step = 0.02
mus = seq(step, 0.5, step)

res05 = matrix(sapply(mus, FUN=runtest05),nrow=length(mus),byrow=TRUE)

par(mar=c(4,4,1,1))
xlab = expression(bold(mu))
# power for balanced data
plot(mus, res05[,1], ylim=c(0,1), ylab='Power', xlab=xlab, type='b', lty=3, 
     lwd=1.2, font.lab=2, frame=FALSE, pch=19, las=1)
points(mus, res05[,3], col=4, type='b', lty = 2, lwd = 1.2,  pch = 15)
points(mus, res05[,2], col=2, type='b', lty = 1, lwd = 1.2,  pch = 17)
grid(lwd=1)
legend("topleft", legend=c("Accuracy Test", "AUC Test A", "AUC Test BM"), 
       col=c(1, 4, 2), lty=c(3,2,1), lwd=1.2, bty="n",  y.intersp=2, pch=c(19,15,17))
dev.copy2pdf(file="power.pdf", width = 7, height = 4.2 )

res02 = matrix(sapply(mus, FUN=runtest02),nrow=length(mus),byrow=TRUE)

# power for unbalanced data
plot(mus, res02[,1], ylim=c(0,1), ylab='Power', xlab=xlab, type='b', lty=3, 
     lwd=1.2, font.lab=2, frame=FALSE, pch=19, las=1)
points(mus, res02[,3], col=4, type='b', lty = 2, lwd = 1.2,  pch = 15)
points(mus, res02[,2], col=2, type='b', lty = 1, lwd = 1.2,  pch = 17)
grid(lwd=1)
legend("topleft", legend=c("Accuracy Test", "AUC Test A", "AUC Test BM"), 
       col=c(1, 4, 2), lty=c(3,2,1), lwd=1.2, bty="n",  y.intersp=2, pch=c(19,15,17))
dev.copy2pdf(file="power8.pdf", width = 7, height = 4.2 )

# to save workspace
if(redoexp) {
  redoexp = FALSE 
  save.image("power.RData")
}

