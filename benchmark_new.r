
library("mlr")
library("OpenML")
library("mlrMBO")

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

getregrLearnerForMBOTuning = function(){
  
  learner = makeLearner("regr.randomForest", predict.type = "se")
  learner = makeImputeWrapper(learner, classes = list(numeric = imputeMedian(), factor = imputeMode()))
  
  return (learner)
}

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

getMultiplexer = function(){
  # SVM classifier
  lrn1 = makeLearner("classif.svm") #, predict.type="prob")
  # SVM hyper-parameter search space
  ps1 = makeParamSet(
    makeNumericParam("cost", lower=-15, upper=15, trafo=function(x) 2^x),
    makeDiscreteParam("kernel", values = c("linear","polynomial","radial","sigmoid"), default = "polynomial"),
    makeIntegerParam("degree",lower = 1L,upper = 5L),
    makeNumericParam("coef0",lower = 0, upper = 1),
    makeNumericParam("gamma", lower=-15, upper=15, trafo=function(x) 2^x)
  )
  
  
  # random forest classifier
  lrn2 = makeLearner("classif.randomForest") #, predict.type="prob")
  # random forest hyper-parameter search space
  ps2 = makeParamSet(
    makeIntegerParam("ntree", lower=1L, upper=9L, trafo = function(x) 2^x),
    makeIntegerParam("mtry", lower = 1L, upper = 30L )
  )
  
  #IBk classifier
  lrn3 = makeLearner("classif.IBk")
  ps3 = makeParamSet(
    makeIntegerParam("K", lower=1L, upper=6L, trafo = function(x) 2^x)
  )
  
  #JRip classifier
  lrn4 = makeLearner("classif.JRip")
  ps4 = makeParamSet(
    makeNumericParam("N", lower = 1, upper = 5),
    makeIntegerParam("O", lower = 1L, upper = 5L)
  )
  
  #1-R classifier
  lrn5 = makeLearner("classif.OneR")
  ps5 = makeParamSet(
    makeIntegerParam("B", lower = 1L, upper = 5L, trafo = function(x) 2^x)
  )
  
  #PART decision lists
  lrn6 = makeLearner("classif.PART")
  ps6 = makeParamSet(
    makeIntegerParam("M", lower = 1L, upper = 6L, trafo = function(x) 2^x),
    makeIntegerParam("N", lower = 2L, upper = 5L)
  )
  
  #J48 Decision Trees
  lrn7 = makeLearner("classif.J48")
  ps7 = makeParamSet(
    makeNumericParam("C", lower = 0, upper = 1),
    makeIntegerParam("M", lower = 1L, upper = 6L, trafo = function(x) 2^x)
  )
  
  #Gradient Boosting
  lrn8 = makeLearner("classif.gbm")
  ps8 = makeParamSet(
    makeIntegerParam("n.trees", lower = 100L, upper = 10000L),
    makeIntegerParam("interaction.depth", lower = 1L, upper = 5L),
    makeNumericParam("shrinkage", lower = -15, upper = -4, trafo = function(x) 2^x)
  )
  
  
  #Naive Bayes
  lrn9 = makeLearner("classif.naiveBayes")
  
  #Logistic
  lrn10 = makeLearner("classif.multinom")
  
  #Neural nets
  lrn11 = makeLearner("classif.avNNet")
  ps11 = makeParamSet(
    makeLogicalParam("bag"),
    makeIntegerParam("repeats", lower = 1, upper = 50)
  )
  
  bls = list(lrn1,lrn2,lrn3,lrn4,lrn5,lrn7,lrn8,lrn9,lrn10,lrn11)#lrn6
  lrn = makeModelMultiplexer(bls)
  
  ps = makeModelMultiplexerParamSet(lrn,
                                    classif.svm = ps1,
                                    classif.randomForest = ps2,
                                    classif.IBk = ps3,
                                    classif.JRip = ps4,
                                    classif.OneR = ps5,
                                    #classif.PART = ps6,
                                    classif.J48 = ps7,
                                    classif.gbm = ps8,
                                    classif.avNNet = ps11
  )
  
  return(list(lrn,ps));
}

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------


tuningTask = function(oml.dataset.id, learner, par.set, budget, perf.measures) {
  
  # specify the dataset
  oml.dataset = getOMLDataSet(oml.dataset.id)
  
  # make a check is imputation is needed
  if (any(is.na(oml.dataset$data))) {
    catf(" - Data imputation required ...")
    temp = impute(data = oml.dataset$data, classes = list(numeric = imputeMean(), factor = imputeMode()))
    oml.dataset$data = temp$data
  }
  
  if(typeof(oml.dataset$data[,oml.dataset$target.features])!="factor"){
    oml.dataset$data[,oml.dataset$target.features] = cut(as.numeric(oml.dataset$data[,oml.dataset$target.features]),breaks = 2)
  }
  
  obj = makeClassifTask(data = oml.dataset$data, target = oml.dataset$desc$default.target.attribute)
  obj$mlr.rin = makeResampleDesc("CV", iters = 10L)
  
  #set nn-net weight
  learner$base.learners$classif.avNNet$par.vals$MaxNWts = 5 * (ncol(oml.dataset$data)) + 5 + 1
  
  inner = makeResampleDesc("CV", iters=5)
  outer = makeResampleInstance("CV", iters=10, task=obj)
  
  irace.inner = makeResampleDesc(method = "Holdout", split=0.8)
  
  aux=list()
  
  # Random Search
  ctrl.random = makeTuneControlRandom(maxit = budget)
  
  control.name = class(ctrl.random)[[1]]
  
  cat("Control",control.name)
  
  
  tuned.learner = tuneParams(learner = learner, task = obj, resampling = inner, measures = perf.measures,
                             par.set = par.set, control = ctrl.random, show.info = TRUE)
  
  
  save(tuned.learner, file=paste("tuneparamresult",control.name,oml.dataset.id,sep = ""))
  
  new.learner = makeLearner(tuned.learner$x$selected.learner)
  
  hp = tuned.learner$x[!is.na(tuned.learner$x)]
  
  hp = hp[!names(hp) %in% c('selected.learner')]
  
  names(hp) = gsub(paste(tuned.learner$x$selected.learner,".",sep=""),"",names(hp))
  
  print(hp)
  
  new.learner = setHyperPars(new.learner, par.vals = hp)
  
  res = resample(learner = new.learner,task = obj,resampling = outer, measures = perf.measures,
                 models = TRUE, keep.pred = TRUE, show.info = TRUE)
  
  aux[[1]] = list(tuned.learner,res)
  
  #irace
  ctrl.irace = makeTuneControlIrace(budget = 5*budget)
  
  control.name = class(ctrl.irace)[[1]]
  
  cat("Control",control.name)
  
  
  tuned.learner = tuneParams(learner = learner, task = obj, resampling = irace.inner, measures = perf.measures,
                             par.set = par.set, control = ctrl.irace, show.info = TRUE)
  
  
  save(tuned.learner, file=paste("tuneparamresult",control.name,oml.dataset.id,sep = ""))
  
  new.learner = makeLearner(tuned.learner$x$selected.learner)
  
  hp = tuned.learner$x[!is.na(tuned.learner$x)]
  
  hp = hp[!names(hp) %in% c('selected.learner')]
  
  names(hp) = gsub(paste(tuned.learner$x$selected.learner,".",sep=""),"",names(hp))
  
  print(hp)
  
  new.learner = setHyperPars(new.learner, par.vals = hp)
  
  res = resample(learner = new.learner,task = obj,resampling = outer, measures = perf.measures,
                 models = TRUE, keep.pred = TRUE, show.info = TRUE)
  
  aux[[1]] = list(tuned.learner,res)
  
  #MBO
  mbo.control = getMBOControl(budget = budget);
  mbo.learner = getregrLearnerForMBOTuning();
  #ctrl.mbo = mlr:::makeTuneControlMBO(mbo.control = mbo.control, learner = mbo.learner)
  
  myObjectiveFunction = function(x){    
    
    x=x[!is.na(x)]
    
    print(x)
    
    # modifying the learner
    new.learner = makeLearner(x$selected.learner)
    
    hp = x[!is.na(x)]
    
    hp = hp[!names(hp) %in% c("selected.learner")]
    
    names(hp) = gsub(paste(x$selected.learner,".",sep=""),"",names(hp))
    
    new.learner = setHyperPars(new.learner, par.vals = hp)    
    
    
    
    res = resample(learner=learner, task=obj, resampling=inner, 
                   models=TRUE, show.info = TRUE, measures=perf.measures)
    
    
    print(res)
    
    value = res$aggr[1]
    #print(value)
    return(value)
    
  }
  
  mbo.result = mbo(fun = myObjectiveFunction, learner = mbo.learner, control = mbo.control,
                   show.info = TRUE, par.set=par.set)
  
  save(mbo.result, file=paste("mboresult",oml.dataset.id,sep = ""))
  
  tuned.learner = makeLearner(mbo.result$x$selected.learner)
  
  hp = mbo.result$x[!is.na(mbo.result$x)]
  
  hp = hp[!names(hp) %in% c("selected.learner")]
  
  names(hp) = gsub(paste(mbo.result$x$selected.learner,".",sep=""),"",names(hp))
  
  print(hp)
  
  tuned.learner = setHyperPars(tuned.learner, par.vals=hp)
  
  
  res = resample(learner = tuned.learner, task = obj, resampling = outer, measures = perf.measures,
           models = TRUE, keep.pred = TRUE, show.info = TRUE)
  
  aux[[3]] = list(mbo.result,res)
  
  return(aux)
  
}


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

getMBOControl = function(budget) {
  
  mbo.control = makeMBOControl(iters = budget,
                               init.design.points = 5 * 3 * 11
  )
  mbo.control = setMBOControlInfill(mbo.control, crit = "ei") #cb throws an error that it is not in the list of crit
  mbo.control = setMBOControlInfill(mbo.control, opt = "focussearch",
                                    opt.restarts = 2L, opt.focussearch.maxit = 2L, opt.focussearch.points = 1000L)
  
  return(mbo.control)
}


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

main = function() {
  #sink("benmarkoutput.out")
  
  budget = 1000
  
  multiplexer.params = getMultiplexer();
  
  lrn = multiplexer.params[[1]]
  ps = multiplexer.params[[2]]
  
  performance.measures = list(mmce, acc, timetrain, timepredict, timeboth)
  
  dataset_ids = c(782,685,867,865,875,1013,736,448,924,885)
  
  for(dataset.id in dataset_ids){
    print(paste("Dataset ID:",dataset.id))
    
    output = tuningTask(oml.dataset.id = dataset.id, learner = lrn, par.set = ps, budget = budget, perf.measures = performance.measures)
    save(output, file=paste("benchmarkoutputs/new_output_",dataset.id,sep=""))
    print(output)
  }
  
  #sink()
}


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

main()

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
