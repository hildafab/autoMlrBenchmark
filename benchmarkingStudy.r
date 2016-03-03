
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
  
  # Random Search
  ctrl.random = makeTuneControlRandom(maxit = budget)
  
  #irace
  ctrl.irace = makeTuneControlIrace(budget = 5*budget)
  
  #MBO
  mbo.control = getMBOControl(budget = budget);
  mbo.learner = getregrLearnerForMBOTuning();
  ctrl.mbo = mlr:::makeTuneControlMBO(mbo.control = mbo.control, learner = mbo.learner)
  
  # List of tuning controls
  ctrls = list(ctrl.random, ctrl.mbo, ctrl.irace)
  
  inner = makeResampleDesc("CV", iters=5)
  outer = makeResampleInstance("CV", iters=10, task=obj)
  
  # Calling tuning techniques (for each tuning control ... )
  aux = lapply(ctrls, function(ct) {
    
    print(paste("Control",toString(class(ct))))
    
    tuned.learner = makeTuneWrapper(learner=learner, resampling=inner, par.set=par.set,
                                    control=ct, show.info=FALSE)
    
    res = resample(learner=tuned.learner, task=obj, resampling=outer,
                   extract=getTuneResult, models=TRUE, show.info = FALSE,
                   measures=perf.measures)
    
    print(res)
    
    return(res)
  })
  
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
  
  dataset_ids = c(4544,1475,44,1041,28,1491,1492,1493,16,14,1039,458)
  
  for(dataset.id in dataset_ids){
    print(paste("Dataset ID:",dataset.id))
    
    output = tuningTask(oml.dataset.id = dataset.id, learner = lrn, par.set = ps, budget = budget, perf.measures = performance.measures)
    save(output, file=paste("benchmarkoutputs/output_",dataset.id,sep=""))
    print(output)
  }
  
  #sink()
}


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

main()

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
