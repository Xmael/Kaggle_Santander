xgb.params=list(       objective="binary:logistic", #binary:logistic (clasf. binaria) o multi:softprob (clasf. multiclase)
                       booster = "gbtree", #gbree o gblinear
                       nthread = 8, #numero de hilo de ejecución
                       set.seed = 123456789,
                       silent = 0,
                       colsample_bytree = 0.6,
                       subsample = 0.8,
                       lambda = 0.8,
                       maximize = F,
                       nrounds=500,
                       eta=0.04,
                       max_depth=6)

h2orf.params=list(
  nthreads=7,
  max_mem_size = "24G",
  max_depth = 15,
  ntrees=50,
  mtries=-1,
  binomial_double_trees = T,
  balance_classes = T
)

h2odeep.params=list(
  nthreads=7,
  max_mem_size="24G"
  )

CV=function(train,test,clase,nfolds,part,algoritmo,params,metrica,multiclass,probabilities){
  library(caret)
  library(AUC)
  library(knitr)
  indice.clase=match(clase,colnames(train))
  folds=createDataPartition(y=train[,indice.clase],times=nfolds,p=part) #Crea particiones
  if(metrica=="auc"){
    evalua=function(pred,test){
      if(!is.factor(test[,indice.clase])){
        test[,indice.clase]=as.factor(test[,indice.clase])
        levels(test)=c(0,1)
      }
      metrica=auc(roc(predictions = pred,labels = test[,indice.clase]),min=0,max=1)
      return(metrica)
    }
  }else if(metrica=="logloss" | metrica=="multilogloss"){
    evalua = function(pred,act){ #Es la multilogloss que también puede valer como logloss
      eps = 1e-15;
      nr = length(pred)
      act=cbind(act)
      pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)
      pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
      ll = sum(act*log(pred) + (1-act)*log(1-pred))
      ll = ll * -1/(nrow(act))
      return(ll)
    }
  }

  #####XGBOOST######
  if(algoritmo=="xgboost"){
    library(xgboost)
    if(params$objective=="multi:softprob"){ #Para multiclase hay que añadir el número de clases
      params$num_class=length(unique(train[,indice.clase]))-1
    }

    #Metricas
    if(metrica=="auc"){
      params$eval_metric="auc"
    }else if(metrica=="logloss"){
      params$eval_metric="logloss"
    }else if(metrica=="multilogloss"){
      params$eval_metric="mlogloss"
    }

    cv=lapply(1:nfolds,function(i){
      test=train[-folds[[i]],]
      xtrain=xgb.DMatrix(data=data.matrix(train[folds[[i]],-indice.clase]),label=train[folds[[i]],indice.clase])
      xtest=xgb.DMatrix(data=data.matrix(train[-folds[[i]],-indice.clase]),label=train[-folds[[i]],indice.clase])
      model=xgboost(data=xtrain,params=params,nrounds=params$nrounds)
      pred=predict(model,xtest)
      metrica=evalua(pred,test)
      return(metrica)
    })
  }

  ####H2O RANDOM FOREST#####
  if(algoritmo=="h2orf"){
    library(h2o)
    h2o.shutdown()
    h2o.init(ip = "localhost",port=54321,startH2O = T,nthreads = h2orf.params$nthreads,max_mem_size = h2orf.params$max_mem_size)
    train[,indice.clase]=as.factor(train[,indice.clase])
    cv=lapply(1:nfolds,function(i){
      test=train[-folds[[i]],]
      htrain=as.h2o(train[folds[[i]],],destination_frame = "train")
      htest=as.h2o(train[-folds[[i]],],destination_frame = "test")
      model=h2o.randomForest(x=c(1:ncol(train))[-indice.clase],y=indice.clase,training_frame = "train",
                             ntrees = h2orf.params$ntrees, max_depth = h2orf.params$max_depth,
                             mtries = h2orf.params$mtries, binomial_double_trees = h2orf.params$binomial_double_trees)
      pred=predict(model,htest)
      metrica=evalua(as.data.frame(pred)[,3],test)
      return(metrica)
    })
    }

    ####H2O DEEP LEARNING
    if(algoritmo=="h2odeep"){
      library(h2o)
      h2o.shutdown()
      h2o.init(ip = "localhost",port=54321,startH2O = T,nthreads = h2odeep.params$nthreads,max_mem_size = h2odeep.params$max_mem_size)
      train[,indice.clase]=as.factor(train[,indice.clase])
      cv=lapply(1:nfolds,function(i){
        test=train[-folds[[i]],]
        htrain=as.h2o(train[folds[[i]],],destination_frame = "train")
        htest=as.h2o(train[-folds[[i]],],destination_frame = "test")
        model=h2o.deep
    }

  resumen=data.frame(rbind(unlist(cv)),mean(unlist(cv)),sd(unlist(cv)))
  colnames(resumen)=c(paste(1:nfolds,"-folds",sep=""),"Mean","SD")
  kable(resumen)
}
