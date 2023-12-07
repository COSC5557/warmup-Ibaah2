#######################################################################
          # Sample of a Simple Machine Learning Algorithm #
#######################################################################

task = tsk("penguins")   # Data

split = partition(task)  # Split the data set into train and test splits
learner = lrn("classif.rpart") # Instantiate the learner "Decision Forest"

learner$train(task, row_ids = split$train) # Fit the learner on the train dataset
learner$model

prediction = learner$predict(task, row_ids = split$test) # Make a prediction on the test dataset.
prediction


prediction$score(msr("classif.acc")) # Get the prediction score on the test data

#######################################################################
           # Practical Machine Learning - Model Selection #
#######################################################################

mlr_learners # Ouputs all the machine learning algorithms "learners" in mlr3

# Output - There are 46 learners in mlr3

# <DictionaryLearner> with 46 stored values
#Keys: classif.cv_glmnet, classif.debug, classif.featureless, classif.glmnet, classif.kknn, classif.lda,
#classif.log_reg, classif.multinom, classif.naive_bayes, classif.nnet, classif.qda, classif.ranger, classif.rpart,
#classif.svm, classif.xgboost, clust.agnes, clust.ap, clust.cmeans, clust.cobweb, clust.dbscan, clust.diana,
#clust.em, clust.fanny, clust.featureless, clust.ff, clust.hclust, clust.kkmeans, clust.kmeans, clust.MBatchKMeans,
#clust.mclust, clust.meanshift, clust.pam, clust.SimpleKMeans, clust.xmeans, regr.cv_glmnet, regr.debug,
#regr.featureless, regr.glmnet, regr.kknn, regr.km, regr.lm, regr.nnet, regr.ranger, regr.rpart, regr.svm,
#regr.xgboost)


install.packages("mlr3verse")

library(mlr3verse)

#############################################################################

                               # Regression

#############################################################################

data <- read.csv("C:/Users/user 1/Documents/Fall _ Spring 2022-2023/Course_Work/Fall 2023/Practical ML/Warmup/50_Startups.csv")
data

new_data <- data[, c("R.D.Spend", "Administration", "Marketing.Spend", "Profit")]
new_data


#############################################################################
              # Decision Tree Regression using rpart #
#############################################################################

task1 = TaskRegr$new(id = "mydata", backend = new_data, target = "Profit")

lrn_rpart1 = lrn("regr.rpart")

splits = partition(task1)

lrn_rpart1$train(task1, splits$train)

prediction1 = lrn_rpart1$predict(task1, splits$test)

# Evaluating the regression model
measures1 = msrs(c("regr.mse", "regr.mae"))

prediction1$score(measures1)

#############################################################################
                  # Linear Regression using regr.lm #
#############################################################################

task2 = TaskRegr$new(id = "mydata", backend = new_data, target = "Profit")

lrn_rpart2 = lrn("regr.lm")

splits = partition(task2)

lrn_rpart2$train(task2, splits$train)

prediction2 = lrn_rpart2$predict(task2, splits$test)

# Evaluating the regression model
measures2 = msrs(c("regr.mse", "regr.mae"))

prediction2$score(measures2)


#############################################################################
                  # KNN Regression using regr.kknn #
#############################################################################
install.packages("kknn")
library(kknn)

task3 = TaskRegr$new(id = "mydata", backend = new_data, target = "Profit")

lrn_rpart3 = lrn("regr.kknn")

splits = partition(task3)

lrn_rpart3$train(task3, splits$train)

prediction3 = lrn_rpart3$predict(task3, splits$test)

# Evaluating the regression model
measures3 = msrs(c("regr.mse", "regr.mae"))

prediction3$score(measures3)

#############################################################################
                    # SVM Regression using regr.svm #
#############################################################################

task4 = TaskRegr$new(id = "mydata", backend = new_data, target = "Profit")

lrn_rpart4 = lrn("regr.svm")

splits = partition(task4)

lrn_rpart4$train(task4, splits$train)

prediction4 = lrn_rpart4$predict(task4, splits$test)

# Evaluating the regression model
measures4 = msrs(c("regr.mse", "regr.mae"))

prediction4$score(measures4)


#############################################################################
               # Neural Net Regression using regr.nnet #
#############################################################################

task5 = TaskRegr$new(id = "mydata", backend = new_data, target = "Profit")

lrn_rpart5 = lrn("regr.nnet")

splits = partition(task5)

lrn_rpart5$train(task5, splits$train)

prediction5 = lrn_rpart5$predict(task5, splits$test)

# Evaluating the regression model
measures5 = msrs(c("regr.mse", "regr.mae"))

prediction5$score(measures5)



#############################################################################

                          # Classification

#############################################################################


data1 <- read.csv("C:/Users/user 1/Documents/Fall _ Spring 2022-2023/Course_Work/Fall 2023/Practical ML/Warmup/Data.csv")
data1

#classif.cv_glmnet, classif.debug, classif.featureless, classif.glmnet, classif.kknn, classif.lda,
#classif.log_reg, classif.multinom, classif.naive_bayes, classif.nnet, classif.qda, classif.ranger, classif.rpart,
#classif.svm, classif.xgboost,

#############################################################################
                # Decision Tree Classification using rpart #
#############################################################################

data1$Class <- as.factor(data1$Class)
str(data1)
task_1 = TaskClassif$new(id = "mydata", backend = data1, target = "Class") 

lrn_rpart_1 = lrn("classif.rpart", predict_type = "prob")

splits = partition(task_1)

lrn_rpart_1$train(task_1, splits$train)

prediction_1 = lrn_rpart_1$predict(task_1, splits$test)

# Evaluating the Classification model
measures_1 = msrs(c("classif.acc", "classif.logloss", "classif.mbrier"))

prediction_1$score(measures_1)

prediction_1$confusion


#############################################################################
              # Logistic Regression using log_reg #
#############################################################################

lrn_logreg_2 = lrn("classif.log_reg", predict_type = "prob")

splits = partition(task_1)

lrn_logreg_2$train(task_1, splits$train)

prediction_2 = lrn_logreg_2$predict(task_1, splits$test)

# Evaluating the Classification model
measures_2 = msrs(c("classif.acc", "classif.logloss", "classif.mbrier"))

prediction_2$score(measures_2)

prediction_2$confusion


#############################################################################
              # SVM Classification using svm #
#############################################################################

lrn_svm_2 = lrn("classif.svm", predict_type = "prob")

splits = partition(task_1)

lrn_svm_2$train(task_1, splits$train)

prediction_3 = lrn_svm_2$predict(task_1, splits$test)

# Evaluating the Classification model
measures_3 = msrs(c("classif.acc", "classif.logloss", "classif.mbrier"))

prediction_3$score(measures_3)

prediction_3$confusion

#############################################################################
              # Random Forest Classification using ranger#
#############################################################################
install.packages("ranger")
library(ranger)

lrn_ranger = lrn("classif.ranger", predict_type = "prob")

splits = partition(task_1)

lrn_ranger$train(task_1, splits$train)

prediction_4 = lrn_ranger$predict(task_1, splits$test)

# Evaluating the Classification model
measures_4 = msrs(c("classif.acc", "classif.logloss", "classif.mbrier"))

prediction_4$score(measures_4)

prediction_4$confusion


#############################################################################
            # KNN Classification using kknn#
#############################################################################


lrn_kknn_2 = lrn("classif.kknn", predict_type = "prob")

splits = partition(task_1)

lrn_kknn_2$train(task_1, splits$train)

prediction_5 = lrn_kknn_2$predict(task_1, splits$test)

# Evaluating the Classification model
measures_5 = msrs(c("classif.acc", "classif.logloss", "classif.mbrier"))

prediction_5$score(measures_5)

prediction_5$confusion


#############################################################################
               # Naive Bayes Classification using naive_bayes#
#############################################################################


lrn_naive_bayes_2 = lrn("classif.naive_bayes", predict_type = "prob")

splits = partition(task_1)

lrn_naive_bayes_2$train(task_1, splits$train)

prediction_6 = lrn_naive_bayes_2$predict(task_1, splits$test)

# Evaluating the Classification model
measures_6 = msrs(c("classif.acc", "classif.logloss", "classif.mbrier"))

prediction_6$score(measures_6)

prediction_6$confusion



############################################################################

                        # Discussion #

############################################################################

# In this model selection problem, I did both regression and classification.

# In the regression task, I applied five leaeners on the dataset. I then compared the
# model performance using mean square error (mse) and mean absolute error (mae).
# A summary of the performance of the models is shown below:

#           Model                   MAE                MSE
#     Decision Tree                 2.5             6.19-08
#     Linear Regression             0.9             1.84-08
#     KNN Regression                1.2             2.7-08
#     SVM Regression                1.4             5.14-08
#     Neural Net Regression         3.1             1.33-09

# From the result of the models, though the models seem not to perform well on 
# the dataset (higher MAE and MSE), Linear regression performed well on the dataset 
# compared to the other models.

# In the classification task, I applied six models on the dataset. The models were 
# compared using accuracy score, log_loss, and mbrier. Again, the confusion matrix was 
# also used. The summary of the models performances are shown below:

#           Model               Accuracy(%)        Log_Loss        Mbrier
#     Decision Tree               0.9292            0.2719         0.1346
#     Logistic Regression         0.9735            0.0811         0.0415
#     SVM Classification          0.9735            0.0811         0.0415         
#     Random Forest               0.9690            0.1049         0.0538
#     KNN Classification          0.9779            0.2053         0.0403
#     Naive Bayes                 0.9469            1.2907         0.1023

# From the above result, we can see that KNN performed well on accuracy and mbrier
# when compared to all the other models, however, logistic regression and SVM achieved
# the lowest log_loss. Based on the result, I will conclude that KNN Classification 
# is the best model.

####################################
            # References #
####################################

# 1. Lang et al., (2019). mlr3: A modern object-oriented machine learning framework in R. Journal of Open Source Software, 
#    4(44),1903, https://doi.org/10.21105/joss.01903.

# 2. Kotthoff L, Sonabend R, Foss N, Bischl B. (2024). Introduction and Overview. In Bischl B, Sonabend R, Kotthoff L, Lang M, 
#    (Eds.), Applied Machine Learning Using mlr3 in R. CRC Press. https://mlr3book.mlr-org.com/introduction_and_overview.html.

# 3. Foss N, Kotthoff L. (2024). Data and Basic Modeling. In Bischl B, Sonabend R, Kotthoff L, Lang M, (Eds.), 
#    Applied Machine Learning Using mlr3 in R. CRC Press. https://mlr3book.mlr-org.com/data_and_basic_modeling.html.