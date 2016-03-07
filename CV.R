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
  nthreads=4,
  max_mem_size="15G",
  hidden=c(370), #(80,40,20)
  seed=1234,
  epochs = 10, #20
  adaptive_rate = F,
  score_validation_samples = 1500,
  score_training_samples = round(nrow(train)*0.5),
  train_samples_per_iteration= 0,
  activation="RectifierWithDropout",
  momentum_start = 0.2,
  momentum_stable = 0.99,
  momentum_ramp = 100,
  rate =0.001, #0.005
  l2 = 0.0002, #0.002
  l1 = 0.0001,
  hidden_dropout_ratios = c(0.5), #c(0.5,0.5,0.5)
  initial_weight_distribution="Normal",
  distribution="multinomial",
  loss="CrossEntropy",
  balance_classes=T,
  fast_mode = F,
  stopping_rounds = 10,
  stopping_metric= "MSE",
  quiet_mode=F,
  nesterov_accelerated_gradient=T
  )

CV=function(train,test,clase,nfolds,algoritmo,params,metrica,multiclass,probabilities){
  library(caret)
  library(knitr)
  indice.clase=match(clase,colnames(train))
  folds=createFolds(y=train[,indice.clase],k = nfolds,list = T) #Crea particiones
  if(metrica=="auc"){
    library(AUC)
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
      xtrain=xgb.DMatrix(data=data.matrix(train[-folds[[i]],-indice.clase]),label=train[-folds[[i]],indice.clase])
      xtest=xgb.DMatrix(data=data.matrix(train[folds[[i]],-indice.clase]),label=train[folds[[i]],indice.clase])
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
      test=train[folds[[i]],]
      htrain=as.h2o(train[-folds[[i]],],destination_frame = "train")
      htest=as.h2o(train[folds[[i]],],destination_frame = "test")
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
        test=train[folds[[i]],]
        htrain=as.h2o(train[-folds[[i]],],destination_frame = "train")
        htest=as.h2o(train[folds[[i]],],destination_frame = "test")
        model=h2o.deeplearning(x=colnames(train)[c(1:ncol(train))[-indice.clase]],y=colnames(train)[indice.clase],training_frame = "train",
                               hidden=h2odeep.params$hidden,
                               seed=h2odeep.params$seed,
                               epochs = h2odeep.params$epochs,
                               adaptive_rate = h2odeep.params$adaptive_rate,
                               score_validation_samples = h2odeep.params$score_validation_samples,
                               score_training_samples = h2odeep.params$score_training_samples,
                               train_samples_per_iteration= h2odeep.params$train_samples_per_iteration,
                               activation=h2odeep.params$activation,
                               momentum_start = h2odeep.params$momentum_start,
                               momentum_stable = h2odeep.params$momentum_stable,
                               momentum_ramp = h2odeep.params$momentum_ramp,
                               rate =h2odeep.params$rate, 
                               l2 = h2odeep.params$l2, 
                               l1 = h2odeep.params$l1,
                               hidden_dropout_ratios = h2odeep.params$hidden_dropout_ratios, 
                               initial_weight_distribution=h2odeep.params$initial_weight_distribution,
                               distribution=h2odeep.params$distribution,
                               loss=h2odeep.params$loss,
                               balance_classes=h2odeep.params$balance_classes,
                               fast_mode = h2odeep.params$fast_mode,
                               stopping_rounds = h2odeep.params$stopping_rounds,
                               stopping_metric= h2odeep.params$stopping_metric,
                               quiet_mode=h2odeep.params$quiet_mode,
                               nesterov_accelerated_gradient=h2odeep.params$nesterov_accelerated_gradient)
        pred=predict(model,htest)
        metrica=evalua(as.data.frame(pred)[,3],test)
        return(metrica)
    })
    }

  resumen=data.frame(rbind(unlist(cv)),mean(unlist(cv)),sd(unlist(cv)))
  colnames(resumen)=c(paste(1:nfolds,"-folds",sep=""),"Mean","SD")
  kable(resumen)
}
