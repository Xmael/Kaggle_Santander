library(kknn)
library(randomForest)
library(caret)
library(e1071)
library(xgboost)
library(AUC)
library(stringr)
library(h2o)
library(foreign) #Solo para cargar el conjunto de datos de prueba en formato arff
setwd("~/Kaggle/Santander")

df=read.arff("phoneme.dat")

indices=createDataPartition(df$Class,times=1,p=0.8)
train=df[indices[[1]],]
test=df[-indices[[1]],]
folds=createFolds(y=train$Class,k=5,list=T)
learners_train=NULL
learners_test=NULL
modelos=NULL

añade=function(stacks,learners_train,learners_test,modelos,algoritmo,parametros){
  aux=rep(NA,nrow(train))
  for(i in 1:5){
    aux[folds[[i]]]=stacks[[i]][[1]]
  }
  cat("Metrica\n")
  print(mean(sapply(1:5,function(i)stacks[[i]][[3]])))
  lee=readline(prompt = "Añadir al stack? [Y]/[N]: ")
  if(str_to_upper(lee)=="Y" | lee==""){
    learners_train=cbind(learners_train,aux)
    learners_test=cbind(learners_test,apply(sapply(1:5,function(i)stacks[[i]][[2]]),1,mean))
    print(head(learners_test))
    modelos=rbind(modelos,cbind(algoritmo=algoritmo,parametros=parametros))
  }else{
    print("Modelo rechazado")
  }
  colnames(learners_train)[ncol(learners_train)]=algoritmo
  colnames(learners_test)[ncol(learners_test)]=algoritmo
  assign(x = 'learners_train',value = learners_train,envir=.GlobalEnv)
  assign(x = 'learners_test',value = learners_test,envir=.GlobalEnv)
  assign(x = 'modelos',value = modelos,envir=.GlobalEnv)
}

####KNN
stack_knn=function(train,k){
  stacks=lapply(1:5,function(i,train,k){
    print(i)
    train_stack=train[-folds[[i]],]
    test_stack=train[folds[[i]],]
    model=kknn(formula=Class~.,train=train_stack,test=test_stack,k=k,distance=2)
    pred_train=model$prob
    model=kknn(formula=Class~.,train=train_stack,test=test,k=5,distance=2)
    pred_test=model$prob
    metrica=auc(roc(pred_train[,2],test_stack[,ncol(test_stack)]))
    return(list(pred_train[,2],pred_test[,2],metrica))
  },train,k)
  algoritmo="knn"
  parametros=paste("k=",k,sep='')
  añade(stacks,learners_train,learners_test,modelos,algoritmo,parametros)
}

#RANDOM FOREST
stack_rf=function(train,ntree){
  stacks=lapply(1:5,function(i,train,ntree){
    print(i)
    train_stack=train[-folds[[i]],]
    test_stack=train[folds[[i]],]
    model=randomForest(Class~.,data=train_stack,ntree=ntree,do.trace=T)
    pred_train=predict(model,test_stack[,-ncol(test_stack)],type="prob")
    pred_test=predict(model,test[,-ncol(test)],type="prob")
    metrica=auc(roc(pred_train[,2],test_stack[,ncol(test_stack)]))
    return(list(pred_train[,2],pred_test[,2],metrica))
  },train,ntree)
  algoritmo="rf"
  parametros=paste("ntree=",ntree,sep='')
  añade(stacks,learners_train,learners_test,modelos,algoritmo,parametros)
}
  
  
  
#SVM
stack_svm=function(train,kernel){
  stacks=lapply(1:5,function(i,train,kernel){
    print(i)
    train_stack=train[-folds[[i]],]
    test_stack=train[folds[[i]],]
    model=svm(x=train_stack[,-ncol(train_stack)],y=train_stack[,ncol(train_stack)],kernel="radial",probability=T,scale=F)
    pred_train=predict(model,test_stack[,-ncol(test_stack)],probability=T)
    pred_test=predict(model,test[-ncol(test)],probability=T)
    metrica=auc(roc(attr(pred_train,"probabilities")[,2],test_stack[,ncol(test_stack)]))
    return(list(attr(pred_train,"probabilities")[,2],attr(pred_test,"probabilities")[,2],metrica))
  },train,kernel)
  algoritmo="svm"
  parametros=paste("kernel=",kernel,sep='')
  añade(stacks,learners_train,learners_test,modelos,algoritmo,parametros)
}

params_xgboost=list(nthred=1,
            objective = "binary:logistic",
            booster="gbtree",
            max_depth=10,
            set.seed=123456789,
            colsample_bytree = 0.6,
            subsample = 0.8,
            lambda = 1,
            maximize = T,
            eval_metric = "auc")

stack_xgboost=function(train,eta){
  stacks=lapply(1:5,function(i,train,eta){
    print(i)
    train_stack=xgb.DMatrix(data=data.matrix(train[-folds[[i]],-ncol(train)]),label=as.numeric(levels(train$Class[-folds[[i]]]))[train$Class[-folds[[i]]]])
    test_stack=xgb.DMatrix(data=data.matrix(train[folds[[i]],-ncol(train)]),label=as.numeric(levels(train$Class[folds[[i]]]))[train$Class[folds[[i]]]])
    model=xgb.train(params=params_xgboost,data=train_stack,nrounds = 50,eta=eta)
    pred_train=predict(model,test_stack)
    pred_test=predict(model,xgb.DMatrix(data=data.matrix(test)))
    metrica=auc(roc(pred_train,train[folds[[i]],ncol(train)]))
    return(list(pred_train,pred_test,metrica))
  },train,eta)
  algoritmo="xgboost"
  parametros=paste("eta=",eta,sep='')
  añade(stacks,learners_train,learners_test,modelos,algoritmo,parametros)
} 

h2o.init(ip = "localhost",port = 54321,nthreads = 2,max_mem_size = "4G")

#H2O Random Forests
stack_h2orf=function(train,ntrees){
  stacks=lapply(1:5,function(i,train,ntrees){
    print(i)
    train_stack=as.h2o(train[-folds[[i]],],destination_frame = "train_stack")
    test_stack=as.h2o(train[folds[[i]],],destination_frame = "test_stack")
    model=h2o.randomForest(x=setdiff(colnames(train_stack),colnames(train_stack)[ncol(train_stack)]),
                           y=colnames(train_stack)[ncol(train_stack)],
                           training_frame = "train_stack",
                           ntrees=50,
                           sample_rate=0.7,
                           mtries=-1,
                           stopping_metric = "AUC",
                           stopping_rounds = 10,
                           binomial_double_trees=T,
                           balance_classes = T,
                           seed=987654321
                           )
    pred_train=as.data.frame(predict(model,test_stack))[3]
    pred_test=as.data.frame(predict(model,as.h2o(test,destination_frame = "test")))[3]
    metrica=auc(roc(pred_train[,1],train[folds[[i]],ncol(train)]))
    return(list(as.numeric(pred_train[,1]),as.numeric(pred_test[,1]),metrica))
  },train,ntrees)
  algoritmo="h2orf"
  parametros=paste("ntrees=",ntrees,sep='')
  añade(stacks,learners_train,learners_test,modelos,algoritmo,parametros)
} 

#H2O deep learning
stack_h2odeep=function(train,activation,initial_weight_distribution,hidden,hidden_dropout_ratios,epochs,rate,l2){
  stacks=lapply(1:5,function(i,train,activation,initial_weight_distribution,hidden,hidden_dropout_ratios,epochs,rate,l2){
    print(i)
    train_stack=as.h2o(train[-folds[[i]],],destination_frame = "train_stack")
    test_stack=as.h2o(train[folds[[i]],],destination_frame = "test_stack")
    model=h2o.deeplearning(x=setdiff(colnames(train_stack),colnames(train_stack)[ncol(train_stack)]),
                           y=colnames(train_stack)[ncol(train_stack)],
                           training_frame = "train_stack",
                           activation =activation,
                           hidden = hidden,
                           hidden_dropout_ratios = hidden_dropout_ratios,
                           epochs=epochs,
                           adaptive_rate = F,
                           initial_weight_distribution=initial_weight_distribution,
                           loss="CrossEntropy",
                           distribution="AUTO",
                           l2=l2,
                           rate=rate,
                           momentum_start = 0.5,
                           momentum_stable=0.99,
                           momentum_ramp=200,
                           nesterov_accelerated_gradient = T,
                           stopping_rounds = 10,
                           stopping_metric = "AUC",
                           balance_classes = T,
                           fast_mode=F,
                           seed = 987654321
    )
    pred_train=as.data.frame(predict(model,test_stack))[3]
    pred_test=as.data.frame(predict(model,as.h2o(test,destination_frame = "test")))[3]
    metrica=auc(roc(pred_train[,1],train[folds[[i]],ncol(train)]))
    return(list(as.numeric(pred_train[,1]),as.numeric(pred_test[,1]),metrica))
  },train,activation,initial_weight_distribution,hidden,hidden_dropout_ratios,epochs,rate,l2)
  algoritmo="h2odeep"
  parametros=paste("activation=",activation," initial_weight_distribution",initial_weight_distribution," hidden",hidden," hidden_dropout_ratios",hidden_dropout_ratios,
                   " epochs",epochs," rate",rate," l2",l2,sep='')
  añade(stacks,learners_train,learners_test,modelos,algoritmo,parametros)
} 




###Hay que revisar un poco a partir de aquí

#learners_train=as.data.frame(learners_train)
#learners_train$Class=train$Class
learners_train=data.frame(learners_train,train)
#learners_test=as.data.frame(learners_test)
learners_test=data.frame(learners_test,test)
colnames(learners_train)[-ncol(learners_train)]=paste("X",c(1:(ncol(learners_train)-1)),sep="")
colnames(learners_test)=paste("X",c(1:ncol(learners_test)),sep="")

stacking=sapply(1:10,function(i){
  model=randomForest(Class~.,data=learners_train,ntree=50,type="prob")
  pred=predict(model,learners_test,type="prob")
  return(auc(roc(pred[,2],test[,ncol(test)]),min=0,max = 1))
})

nostacking=sapply(1:10,function(i){
  model=randomForest(Class~.,data=train,ntree=50,type="prob")
  pred=predict(model,test[,-ncol(test)],type="prob")
  return(auc(roc(pred[,2],test[,ncol(test)]),min=0,max = 1))
})

print(mean(stacking))
print(mean(nostacking))