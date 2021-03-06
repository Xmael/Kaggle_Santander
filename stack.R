library(kknn)
library(randomForest)
library(caret)
library(e1071)
library(xgboost)
library(AUC)
library(stringr)
library(h2o)
library(Rtsne)
library(foreign) #Solo para cargar el conjunto de datos de prueba en formato arff
setwd("~/Kaggle/Santander")

df=read.arff("magic.dat")

#Hay que cambiar el nombre de la columna clase a class
indice_clase=ncol(df) #Modificar esto si procede
colnames(df)[indice_clase]="class"
df$class=as.factor(df$class)
levels(df$class)[c(which.min(table(df$class)),which.max(table(df$class)))]=c(1,0)
nfolds=5 #Esto lo hacemos para no olvidarnos que en el siguiente nivel el número de particiones es una unidad menos

indices=createDataPartition(df$class,times=1,p=0.8)
train=df[indices[[1]],]
test=df[-indices[[1]],-indice_clase]
folds=createFolds(y=train$class,k=nfolds,list=T)

inicializa=function(){
  learners_train=NULL
  learners_test=NULL
  modelos=NULL
  assign("learners_train",learners_train,envir = .GlobalEnv)
  assign("learners_test",learners_test,envir = .GlobalEnv)
  assign("modelos",modelos,envir = .GlobalEnv)
}

inicializa()

tsne=function(train,test){
  perpvalue=100
    features=train[,-ncol(train)]
    tsne=Rtsne(as.matrix(features), check_duplicates = FALSE, pca = F, 
               perplexity=perpvalue, theta=0.5, dims=2,verbose=T)
    train_tsne= as.data.frame(tsne$Y)
    train_tsne$class=as.factor(train[,ncol(train)])
    features=test
    tsne=Rtsne(as.matrix(features), check_duplicates = FALSE, pca = F, 
               perplexity=perpvalue, theta=0.5, dims=2,verbose = T)
    test_tsne = as.data.frame(tsne$Y)
    return(list(train_tsne,test_tsne))
}

añade=function(nfolds,folds,stacks,learners_train,learners_test,modelos,algoritmo,parametros){
  aux=rep(NA,nrow(train))
  for(i in 1:nfolds){
    aux[folds[[i]]]=stacks[[i]][[1]]
  }
  cat("Metrica\n")
  metrica=mean(sapply(1:nfolds,function(i)stacks[[i]][[3]]))
  print(metrica)
#  lee=readline(prompt = "Añadir al stack? [Y]/[N]: ")
#  if(str_to_upper(lee)=="Y" | lee==""){
    learners_train=cbind(learners_train,aux)
    learners_test=cbind(learners_test,apply(sapply(1:nfolds,function(i)stacks[[i]][[2]]),1,mean))
    print(head(learners_test))
    modelos=rbind(modelos,cbind(algoritmo=algoritmo,parametros=parametros,metrica=metrica))
    colnames(learners_train)[ncol(learners_train)]=algoritmo
    colnames(learners_test)[ncol(learners_test)]=algoritmo
    assign(x = 'learners_train',value = learners_train,envir=.GlobalEnv)
    assign(x = 'learners_test',value = learners_test,envir=.GlobalEnv)
    assign(x = 'modelos',value = modelos,envir=.GlobalEnv)
#  }else{
#    print("Modelo rechazado")
#  }
}

construye=function(learners_train,learners_test){
  lee=readline(prompt = "Quiere añadir el conjunto de características inicial? [Y]/[N]: ")
  if(str_to_upper(lee)=="Y" | str_to_upper(lee)==""){
    learners_train=data.frame(learners_train,train)
    learners_test=data.frame(learners_test,test)
  }else{
    learners_train=as.data.frame(learners_train)
    learners_train$class=train$class
    learners_test=as.data.frame(learners_test)
  }
  colnames(learners_train)[-ncol(learners_train)]=paste("X",c(1:(ncol(learners_train)-1)),sep="")
  colnames(learners_test)=paste("X",c(1:ncol(learners_test)),sep="")
  assign("learners_train",learners_train,envir = .GlobalEnv)
  assign("learners_test",learners_test,envir=.GlobalEnv)
}

####KNN
stack_knn=function(nfolds,train,test,folds,k){
  stacks=lapply(1:nfolds,function(i,train,test,folds,k){
    print(i)
    train_stack=train[-folds[[i]],]
    test_stack=train[folds[[i]],]
    model=kknn(formula=class~.,train=train_stack,test=test_stack,k=k,distance=2)
    pred_train=model$prob
    model=kknn(formula=class~.,train=train_stack,test=test,k=k,distance=2)
    pred_test=model$prob
    metrica=auc(roc(pred_train[,2],test_stack[,ncol(test_stack)]))
    return(list(pred_train[,2],pred_test[,2],metrica))
  },train,test,folds,k)
  algoritmo="knn"
  parametros=paste("k=",k,sep='')
  añade(nfolds,folds,stacks,learners_train,learners_test,modelos,algoritmo,parametros)
}

#RANDOM FOREST
stack_rf=function(nfolds,train,test,folds,ntree){
  stacks=lapply(1:nfolds,function(i,train,test,folds,ntree){
    print(i)
    train_stack=train[-folds[[i]],]
    test_stack=train[folds[[i]],]
    model=randomForest(class~.,data=train_stack,ntree=ntree,do.trace=T)
    pred_train=predict(model,test_stack[,-ncol(test_stack)],type="prob")
    pred_test=predict(model,test,type="prob")
    metrica=auc(roc(pred_train[,2],test_stack[,ncol(test_stack)]))
    return(list(pred_train[,2],pred_test[,2],metrica))
  },train,test,folds,ntree)
  algoritmo="rf"
  parametros=paste("ntree=",ntree,sep='')
  añade(nfolds,folds,stacks,learners_train,learners_test,modelos,algoritmo,parametros)
}

#SVM
stack_svm=function(nfolds,train,test,folds,kernel){
  stacks=lapply(1:nfolds,function(i,train,folds,kernel){
    print(i)
    train_stack=train[-folds[[i]],]
    test_stack=train[folds[[i]],]
    model=svm(x=train_stack[,-ncol(train_stack)],y=train_stack[,ncol(train_stack)],kernel="radial",probability=T,scale=F)
    pred_train=predict(model,test_stack[,-ncol(test_stack)],probability=T)
    pred_test=predict(model,test,probability=T)
    metrica=auc(roc(attr(pred_train,"probabilities")[,2],test_stack[,ncol(test_stack)]))
    return(list(attr(pred_train,"probabilities")[,2],attr(pred_test,"probabilities")[,2],metrica))
  },train,folds,kernel)
  algoritmo="svm"
  parametros=paste("kernel=",kernel,sep='')
  añade(nfolds,folds,stacks,learners_train,learners_test,modelos,algoritmo,parametros)
}

params_xgboost=list(nthred=8,
                    objective = "binary:logistic",
                    booster="gbtree",
                    max_depth=10,
                    set.seed=123456789,
                    colsample_bytree = 0.6,
                    subsample = 0.8,
                    lambda = 1,
                    maximize = T,
                    eval_metric = "auc")

stack_xgboost=function(nfolds,train,test,folds,eta,nrounds){
  stacks=lapply(1:nfolds,function(i,train,test,folds,eta,nrounds){
    print(i)
    train_stack=xgb.DMatrix(data=data.matrix(train[-folds[[i]],-ncol(train)]),label=as.numeric(levels(train$class[-folds[[i]]]))[train$class[-folds[[i]]]])
    test_stack=xgb.DMatrix(data=data.matrix(train[folds[[i]],-ncol(train)]),label=as.numeric(levels(train$class[folds[[i]]]))[train$class[folds[[i]]]])
    model=xgb.train(params=params_xgboost,data=train_stack,nrounds = 50,eta=eta)
    pred_train=predict(model,test_stack)
    pred_test=predict(model,xgb.DMatrix(data=data.matrix(test)))
    metrica=auc(roc(pred_train,train[folds[[i]],ncol(train)]))
    return(list(pred_train,pred_test,metrica))
  },train,test,folds,eta,nrounds)
  algoritmo="xgboost"
  parametros=paste("eta=",eta," nrounds=",sep='')
  añade(nfolds,folds,stacks,learners_train,learners_test,modelos,algoritmo,parametros)
} 

h2o.init(ip = "localhost",port = 54321,nthreads = 2,max_mem_size = "4G")

#H2O Random Forests
stack_h2orf=function(nfolds,train,test,folds,ntrees){
  stacks=lapply(1:nfolds,function(i,train,test,folds,ntrees){
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
  },train,test,folds,ntrees)
  algoritmo="h2orf"
  parametros=paste("ntrees=",ntrees,sep='')
  añade(nfolds,folds,stacks,learners_train,learners_test,modelos,algoritmo,parametros)
} 

#H2O deep learning
stack_h2odeep=function(nfolds,train,test,folds,activation,initial_weight_distribution,hidden,hidden_dropout_ratios,epochs,rate,l2){
  stacks=lapply(1:nfolds,function(i,train,test,folds,activation,initial_weight_distribution,hidden,hidden_dropout_ratios,epochs,rate,l2){
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
  },train,test,folds,activation,initial_weight_distribution,hidden,hidden_dropout_ratios,epochs,rate,l2)
  algoritmo="h2odeep"
  parametros=paste("activation=",activation," initial_weight_distribution",initial_weight_distribution," hidden",hidden," hidden_dropout_ratios",hidden_dropout_ratios,
                   " epochs",epochs," rate",rate," l2",l2,sep='')
  añade(nfolds,folds,stacks,learners_train,learners_test,modelos,algoritmo,parametros)
} 

######Llamadas

stack_knn(nfolds=nfolds,train=train,test=test,folds=folds,k=3)
stack_rf(nfolds=nfolds,train=train,test=test,folds=folds,ntree=50)
stack_svm(nfolds=nfolds,train=train,test=test,folds=folds,kernel="radial")
stack_xgboost(nfolds=nfolds,train=train,test=test,folds=folds,eta=0.5,nrounds=20)
stack_h2orf(nfolds=nfolds,train=train,test=test,folds=folds,ntrees=100)
stack_h2odeep(nfolds=nfolds,train=train,test=test,folds=folds,activation=,initial_weight_distribution=,
              hidden=,hidden_dropout_ratios=,epochs=,rate=,l2=)

inicializa()
stack_knn(nfolds = nfolds,train = train,test = test,folds = folds,k = 3)
stack_knn(nfolds = nfolds,train = train,test = test,folds = folds,k = 7)
stack_knn(nfolds = nfolds,train = train,test = test,folds = folds,k = 13)
stack_rf( nfolds = nfolds,train = train,test = test,folds = folds,ntree = 50)
stack_rf( nfolds = nfolds,train = train,test = test,folds = folds,ntree = 100)
stack_svm(nfolds = nfolds,train = train,test = test,folds = folds,kernel = "radial")
stack_xgboost(nfolds = nfolds,train = train,test = test,folds = folds,eta = 0.5,nrounds = 20)
stack_xgboost(nfolds = nfolds,train = train,test = test,folds = folds,eta = 0.1,nrounds = 50)

learners_train_bak=learners_train
learners_test_bak=learners_test

###Hay que revisar un poco a partir de aquí

construye(learners_train = learners_train, learners_test = learners_test)

#Guardamos el stack del primer nivel

learners_train_1st=learners_train
learners_test_1st=learners_test
modelos_1st=modelos

pfolds=createFolds(y=learners_train_1st$class,k=nfolds-1,list=T)
inicializa()

stack_knn(nfolds=nfolds-1,train=learners_train_1st,test=learners_test_1st,folds=pfolds,k=7)
stack_rf(nfolds=nfolds-1,train=learners_train_1st,test=learners_test_1st,folds=pfolds,ntree=100)
stack_svm(nfolds=nfolds-1,train=learners_train_1st,test=learners_test_1st,folds=pfolds,kernel)
stack_xgboost(nfolds=nfolds-1,train=learners_train_1st,test=learners_test_1st,folds=pfolds,eta=0.5,nrounds = 60)
stack_h2orf(nfolds=nfolds-1,train=learners_train_1st,test=learners_test_1st,folds=pfolds,ntrees)
stack_h2odeep(nfolds=nfolds-1,train=learners_train_1st,test=learners_test_1st,folds=pfolds,activation=,initial_weight_distribution=,
              hidden=,hidden_dropout_ratios=,epochs=,rate=,l2=)

#Es mejor crear un ensemble con las predicciones del segundo nivel

learners_train_2nd=learners_train
learners_test_2nd=learners_test
modelos_2nd=modelos

ensemble=apply(learners_test_2nd,1,mean)

#######
stacking=mclapply(1:10,function(i){
  print(i)
  model=randomForest(class~.,data=learners_train_2nd,ntree=100,type="prob")
  pred=predict(model,learners_test_2nd,type="prob")
  return(auc(roc(pred[,2],df[-indices[[1]],indice_clase]),min=0,max = 1))
},mc.preschedule = TRUE, mc.set.seed = TRUE,
mc.silent = FALSE, mc.cores = getOption("mc.cores", 5L),
mc.cleanup = TRUE, mc.allow.recursive = TRUE)

nostacking=mclapply(1:10,function(i){
  print(i)
  model=randomForest(class~.,data=train,ntree=100,type="prob")
  pred=predict(model,test,type="prob")
  return(auc(roc(pred[,2],df[-indices[[1]],indice_clase]),min=0,max = 1))
},mc.preschedule = TRUE, mc.set.seed = TRUE,
mc.silent = FALSE, mc.cores = getOption("mc.cores", 5L),
mc.cleanup = TRUE, mc.allow.recursive = TRUE)

print(mean(unlist(stacking)))
print(mean(unlist(nostacking)))


cv=mclapply(1:5,function(i){
  model=randomForest(class~.,resumen[[i]][[1]],ntree=50)
  pred=predict(model,resumen[[i]][[2]],type="prob")
  return(auc(roc(pred[,2],resumen[[i]][[2]]$class),min=0,max = 1))
},mc.preschedule = TRUE, mc.set.seed = TRUE,
mc.silent = FALSE, mc.cores = getOption("mc.cores", 5L),
mc.cleanup = TRUE, mc.allow.recursive = TRUE)