---
title: Random Forest
author: Thomas
date: []
slug: random-forest
categories: []
tags:
  - ml
header:
  caption: ''
  image: ''
---

## Random Forests

Random forests are based on a simple idea: *'the wisdom of the crowd'*. Aggregate of the results of multiple predictors gives a better prediction than the best individual predictor. A group of predictors is called an **ensemble**. Thus, this technique is called **Ensemble Learning**. 

In earlier tutorial, you learned how to use **Decision trees** to make a binary prediction. To improve our technique, we can train a group of **Decision Tree classifiers**, each on a different random subset of the train set. To make a prediction, we just obtain the predictions of all individuals trees, then predict the class that gets the most votes. This technique is called **Random Forest**.

We will proceed as follow to train the Random Forest:

- Step 1: Import the data
- Step 2: Train the model
- Step 3: Construct accuracy function
- Step 4: Visualize the model

### Step 1: Import the data

To make sure you have the same dataset as in the tutorial for decision trees, the train test and test set are stored on the internet. You can import them without make any change.


```r
library(dplyr)
data_train <- read.csv("https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/titanic_train.csv") %>%
  select(-1)
data_test <- read.csv("https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/titanic_test.csv") %>%
  select(-1)
```

### Step 2:  Train the model

One way to evaluate the performance of a model is to train it on a number of different smaller datasets and evaluate them over the other smaller testing set. This is called the **F-fold cross-validation** feature. R has a function to randomly split $k$ number of datasets of almost the same size. For example, if $k=9$, the model is evaluated over the nine folder and tested on the remaining test set. This process is repeated until all the subsets have been evaluated. This technique is widely used for model selection, especially when the model has parameters to tune.

Now that we have a way to evaluate our model, we need to figure out how to choose the parameters that generalized best the data.

Random forest chooses a random subset of features and builds many Decision Trees. The model averages out all the predictions of the Decisions trees. 

Random forest has some parameters that can be changed to improved the generalization of the prediction. You will use the function `RandomForest()` to train the model.

Syntax for Random Forest is:

```
RandomForest(formula, ntree=n, mtry=FALSE, maxnodes = NULL)
Arguments:

- Formula: Formula of the fitted model
- ntree: number of trees in the forest
- mtry: Number of candidates draw to feed the algorithm. By default, it is the square of the number of columns.
- maxnodes: Set the maximum amount of terminal nodes in the forest
-	importance=TRUE: Whether independent variables importance in the random forest be assessed
```

*note*: Random forest can be trained on more parameters. You can refer to the [vignette](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf) to see the different parameters.

Tuning a model is very tedious work. There are lot of combination possible  between the parameters. You don't necessarily have the time to try all of them. A good alternative is to let the machine find the best combination for you. There are two methods  available: 

- Random Search 
- Grid Search

We will define both methods but during the tutorial, we will train the model using grid search

### Grid Search

The grid search method is simple, the model will be evaluated over all the combination you pass in the function, using cross-validation. 

For instance, you want to try the model with 10, 20, 30 number of trees and each tree will be tested over a number of `mtry` equals to 1, 2, 3, 4, 5. Then the machine will test 15 different models:


```
##    .mtry ntrees
## 1      1     10
## 2      2     10
## 3      3     10
## 4      4     10
## 5      5     10
## 6      1     20
## 7      2     20
## 8      3     20
## 9      4     20
## 10     5     20
## 11     1     30
## 12     2     30
## 13     3     30
## 14     4     30
## 15     5     30
```


The algorithm will evaluate:

```
RandomForest(formula, ntree=10, mtry=1)
RandomForest(formula, ntree=10, mtry=2)
RandomForest(formula, ntree=10, mtry=3)
RandomForest(formula, ntree=20, mtry=2)
and so on
```
Each time, the random forest experiments with a cross-validation. 
One shortcoming of the grid search is the number of experimentations. It can become very easily explosive when the number of combination is high. To overcome this issue, you can use the random search

### Random Search definition

The big difference between random search and grid search is, random search will not evaluate all the combination of hyperparameter in the searching space. Instead, it will randomly choose combination at every iteration. The advantage is it lower the computational cost. 

### Set the control parameter

You will proceed as follow to construct and evaluate the model:

- Evaluate the model with the default setting
- Find the best number of `mtry`
- Find the best number of `maxnodes`
- Find the best number of `ntrees`
- Evaluate the model on the test dataset

Before you begin with the parameters exploration, you need to install two libraries. 

- `caret`: R machine learning library. If you have install R with `r-essential`. It is already in the library
    - [Anaconda](https://anaconda.org/r/r-caret): `conda install -c r r-caret`
- `e1071`: R machine learning library.
    - [Anaconda](https://anaconda.org/r/r-caret): `conda install -c r r-e1071`

You can import them along with `RandomForest`


```r
library(randomForest)
library(caret)
library(e1071)
```

**Default setting**

K-fold cross validation is controlled by the `trainControl()` function

```
trainControl(method = "cv", number = n, search ="grid")
arguments

- method = "cv": The method used to resample the dataset. 
- number = n: Number of folders to create
- search = "grid": Use the search grid method. For randomized method, use "grid"

Note: You can refer to the vignette to see the other arguments of the function.
```
You can try to run the model with the default parameters and see the accuracy score.

*note*: You will use the same controls during all the tutorial.


```r
# Define the control
trControl <- trainControl(method="cv",
                          number=10,
                          search="grid")
```

You will use `caret` library to evaluate your model. The library has one function called `train()` to evaluate almost all machine learning algorithm. Say differently, you can use this function to train other algorithms. 

The basic syntax is:

```
train(formula, df, method = "rf", metric= "Accuracy", trControl = trainControl(), tuneGrid = NULL)
argument

- `formula`: Define the formula of the algorithm
- `method`: Define which model to train. Note, at the end of the tutorial, there is a list of all the models that can be trained
- `metric` = "Accuracy": Define how to select the optimal model
- `trControl = trainControl()`: Define the control parameters
- `tuneGrid = NULL`: Return a data frame with all the possible combination
```

Let's try the build the model with the default values.


```r
set.seed(1234)
# Run the model
rf_default <- train(survived ~.,
                    data=data_train,
                    method="rf",
                    metric="Accuracy",
                    trControl=trControl)
# Print the results
print(rf_default)
```

```
## Random Forest 
## 
## 836 samples
##   7 predictor
##   2 classes: 'No', 'Yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 753, 752, 753, 752, 752, 752, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.7955823  0.5612256
##    6    0.7787292  0.5341776
##   10    0.7631670  0.5051445
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

Code Explanation 

- `trainControl(method="cv", number=10, search="grid")`: Evaluate the model with a grid search of 10 folder
- `train(...)`: Train a random forest model. Best model is chosen with the accuracy measure. 

The algorithm uses 500 trees and tested three different values of `mtry`: 2, 6, 10. 

The final value used for the model was `mtry = 2` with an accuracy of 0.78. Let's try to get a higher score. 

**Step 2: Search best `mtry`**

You can test the model with values of `mtry` from 1 to 10


```r
set.seed(1234)
tuneGrid <- expand.grid(.mtry=c(1:10))
rf_mtry <- train(survived ~.,
                    data=data_train,
                    method="rf",
                    metric="Accuracy",
                    tuneGrid=tuneGrid,
                    trControl=trControl,
                    importance = TRUE, 
                    nodesize = 14, 
                    ntree = 300)
print(rf_mtry)
```

```
## Random Forest 
## 
## 836 samples
##   7 predictor
##   2 classes: 'No', 'Yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 753, 752, 753, 752, 752, 752, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    1    0.7512622  0.4493869
##    2    0.8087063  0.5879055
##    3    0.8051061  0.5827678
##    4    0.8074871  0.5888521
##    5    0.8086776  0.5919749
##    6    0.8074871  0.5890573
##    7    0.7991107  0.5724776
##    8    0.8014917  0.5774352
##    9    0.8062823  0.5874728
##   10    0.8003012  0.5753901
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

Code Explanation

- `tuneGrid <- expand.grid(.mtry=c(3:10))`: Construct a vector with value from 3:10

The final value used for the model was `mtry = 4.

The best value of `mtry` is stored in: 

```
rf_mtry$bestTune$mtry
```

You can store it and use it when you need to tune the other parameters.


```r
max(rf_mtry$results$Accuracy)
```

```
## [1] 0.8087063
```

```r
best_mtry <- rf_mtry$bestTune$mtry
best_mtry
```

```
## [1] 2
```

**Step 3: Search the best `maxnodes`**

You need to create a loop to evaluate the different values of `maxnodes`. In the following code, you will:

- Create a list
- Create a variable with the best value of the parameter `mtry`; Compulsory
- Create the loop
- Store the current value of `maxnode`
- Summarize the results


```r
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry=best_mtry)
for (maxnodes in c(5:15)) {
  set.seed(1234)
  rf_maxnode <- train(survived ~.,
                   data=data_train,
                   method="rf",
                   metric="Accuracy",
                   tuneGrid=tuneGrid,
                   trControl=trControl, 
                   importance = TRUE,
                   nodesize = 14,
                   maxnodes=maxnodes,
                   ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}  
results_mtry <- resamples(store_maxnode)
summary(results_mtry)
```

```
## 
## Call:
## summary.resamples(object = results_mtry)
## 
## Models: 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 
## Number of resamples: 10 
## 
## Accuracy 
##         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
## 5  0.6190476 0.7291667 0.7605422 0.7512765 0.7850688 0.8214286    0
## 6  0.6428571 0.7410714 0.7725186 0.7572433 0.7850688 0.8095238    0
## 7  0.7142857 0.7507172 0.7951807 0.7823580 0.8065476 0.8433735    0
## 8  0.6666667 0.7500000 0.7904475 0.7811819 0.8313253 0.8333333    0
## 9  0.7380952 0.7552711 0.7891566 0.7823437 0.8065476 0.8214286    0
## 10 0.7261905 0.7582831 0.7831325 0.7847533 0.8154762 0.8433735    0
## 11 0.7261905 0.7910929 0.8132530 0.7991107 0.8208907 0.8333333    0
## 12 0.7380952 0.7791523 0.8072289 0.7967154 0.8214286 0.8333333    0
## 13 0.7500000 0.7672117 0.8083764 0.7967154 0.8208907 0.8333333    0
## 14 0.7380952 0.7910929 0.8072289 0.8003155 0.8214286 0.8433735    0
## 15 0.7380952 0.7910929 0.8144005 0.8051348 0.8303571 0.8433735    0
## 
## Kappa 
##         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
## 5  0.1703704 0.4010000 0.4827780 0.4571191 0.5332083 0.6052632    0
## 6  0.2144638 0.4255725 0.4973450 0.4702368 0.5332083 0.5851852    0
## 7  0.3838631 0.4549183 0.5600369 0.5290173 0.5803170 0.6717371    0
## 8  0.2668329 0.4528536 0.5505108 0.5246700 0.6376286 0.6480921    0
## 9  0.4181360 0.4639628 0.5481472 0.5305634 0.5863434 0.6246608    0
## 10 0.3823529 0.4700053 0.5316585 0.5322930 0.6029475 0.6717371    0
## 11 0.4180723 0.5470809 0.5998407 0.5671280 0.6158468 0.6405868    0
## 12 0.4239401 0.5235007 0.5858807 0.5612607 0.6148865 0.6448655    0
## 13 0.4528536 0.4952796 0.5858678 0.5611099 0.6157794 0.6448655    0
## 14 0.4406780 0.5470809 0.5856454 0.5691846 0.6148865 0.6717371    0
## 15 0.4406780 0.5470809 0.5978788 0.5786701 0.6372479 0.6687135    0
```

Code explanation:

- `store_maxnode <- list()`: The results of the model will be stored in this list
- `expand.grid(.mtry=best_mtry)`:  Use the best value of `mtry`
- `for (maxnodes in c(15:25)) { ... }`: Compute the model with values of `maxnodes` starting from 15 to 25.
- `maxnodes=maxnodes`: For each iteration, `maxnodes` is equal to the current value of  `maxnodes`. i.e 15, 16, 17, ...
- `key <- toString(maxnodes)`: Store as a string variable the value of `maxnode`.
- `store_maxnode[[key]] <- rf_maxnode`: Save the result of the model in the list. 
- `resamples(store_maxnode)`: Arrange the results of the model
- `summary(results_mtry)`: Print the summary of all the combination.

The last value of `maxnode` has the highest accuracy. You can try with higher values to see if you can get a higher score.


```r
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry=best_mtry)
for (maxnodes in c(20:30)) {
  set.seed(1234)
  rf_maxnode <- train(survived ~.,
                   data=data_train,
                   method="rf",
                   metric="Accuracy",
                   tuneGrid=tuneGrid,
                   trControl=trControl, 
                   importance = TRUE,
                   nodesize = 14,
                   maxnodes=maxnodes, 
                   ntree = 300)
  key <- toString(maxnodes)
  store_maxnode[[key]] <- rf_maxnode
}  
results_node <- resamples(store_maxnode)
summary(results_node)
```

```
## 
## Call:
## summary.resamples(object = results_node)
## 
## Models: 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 
## Number of resamples: 10 
## 
## Accuracy 
##         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
## 20 0.7142857 0.7747418 0.8072289 0.7978916 0.8303571 0.8452381    0
## 21 0.7261905 0.7867542 0.8192771 0.8050775 0.8378873 0.8452381    0
## 22 0.7261905 0.7880809 0.8132530 0.8003299 0.8214286 0.8554217    0
## 23 0.7142857 0.7837780 0.8132530 0.8003012 0.8378873 0.8452381    0
## 24 0.7142857 0.7910929 0.8132530 0.8051061 0.8333333 0.8554217    0
## 25 0.7142857 0.7851764 0.8263769 0.8063540 0.8328313 0.8554217    0
## 26 0.7142857 0.7880809 0.8072289 0.8014917 0.8378873 0.8571429    0
## 27 0.7023810 0.7821644 0.8072289 0.8015060 0.8378873 0.8452381    0
## 28 0.7142857 0.7910929 0.8132530 0.8027252 0.8303571 0.8554217    0
## 29 0.7261905 0.7880809 0.8132530 0.8027108 0.8303571 0.8554217    0
## 30 0.7142857 0.7702238 0.8203528 0.8015491 0.8408635 0.8452381    0
## 
## Kappa 
##         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
## 20 0.3956835 0.5038017 0.5858807 0.5640020 0.6327354 0.6717371    0
## 21 0.4180723 0.5314517 0.6098768 0.5781808 0.6516526 0.6717371    0
## 22 0.4180723 0.5409748 0.5998407 0.5699488 0.6186170 0.6955990    0
## 23 0.3956835 0.5340991 0.5919064 0.5688979 0.6511075 0.6717371    0
## 24 0.3956835 0.5480930 0.5972354 0.5795206 0.6396994 0.6955990    0
## 25 0.3956835 0.5382442 0.6220489 0.5823127 0.6391453 0.6955990    0
## 26 0.3956835 0.5409748 0.5851504 0.5717111 0.6535555 0.6888889    0
## 27 0.3735084 0.5296068 0.5856454 0.5725588 0.6541679 0.6717371    0
## 28 0.3956835 0.5484925 0.5928882 0.5740667 0.6336956 0.6955990    0
## 29 0.4236277 0.5425302 0.5967170 0.5748578 0.6327354 0.6955990    0
## 30 0.3956835 0.5089240 0.6117973 0.5735369 0.6546210 0.6717371    0
```

The highest accuracy score is obtained with a value of `maxnode` equals to 22.

**Step 4: Search the best `ntrees`**

Now that you have the best value of `mtry` and `maxnode`, you can tune the number of trees. The method is exactly the same as `maxnode`. 


```r
store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)){
    set.seed(5678)
    rf_maxtrees <- train(survived ~.,
                       data=data_train,
                       method="rf",
                       metric="Accuracy",
                       tuneGrid=tuneGrid,
                       trControl=trControl,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes=24,
                       ntree = ntree)
    key <- toString(ntree)
    store_maxtrees[[key]] <- rf_maxtrees
}  
results_tree <- resamples(store_maxtrees)
summary(results_tree)
```

```
## 
## Call:
## summary.resamples(object = results_tree)
## 
## Models: 250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000 
## Number of resamples: 10 
## 
## Accuracy 
##           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
## 250  0.7261905 0.7761403 0.8035714 0.7931818 0.8151858 0.8554217    0
## 300  0.7380952 0.7761403 0.8035714 0.7979437 0.8203397 0.8554217    0
## 350  0.7380952 0.7761403 0.8095238 0.7991632 0.8273084 0.8554217    0
## 400  0.7380952 0.7791523 0.8035714 0.8003824 0.8243322 0.8674699    0
## 450  0.7380952 0.7837780 0.8095238 0.8015442 0.8243322 0.8554217    0
## 500  0.7380952 0.7837780 0.8035714 0.8027346 0.8273084 0.8554217    0
## 550  0.7380952 0.7761403 0.8035714 0.8003246 0.8203397 0.8554217    0
## 600  0.7500000 0.7791523 0.8095238 0.8051299 0.8273084 0.8554217    0
## 800  0.7380952 0.7761403 0.8095238 0.8027346 0.8273084 0.8554217    0
## 1000 0.7380952 0.7837780 0.8073461 0.8027056 0.8214286 0.8554217    0
## 2000 0.7380952 0.7791523 0.8095238 0.8039395 0.8273084 0.8554217    0
## 
## Kappa 
##           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
## 250  0.3823529 0.5242512 0.5774677 0.5542996 0.5997614 0.6927822    0
## 300  0.4061697 0.5242512 0.5794569 0.5636594 0.6120353 0.6927822    0
## 350  0.4061697 0.5242512 0.5892421 0.5661675 0.6220830 0.6927822    0
## 400  0.4061697 0.5308812 0.5780106 0.5688971 0.6152676 0.7196807    0
## 450  0.4061697 0.5399732 0.5892421 0.5709407 0.6152676 0.6927822    0
## 500  0.4061697 0.5399732 0.5774677 0.5736169 0.6220830 0.6927822    0
## 550  0.4061697 0.5242512 0.5774677 0.5683979 0.6137421 0.6927822    0
## 600  0.4302326 0.5308812 0.5917838 0.5792288 0.6220830 0.6927822    0
## 800  0.4061697 0.5242512 0.5912409 0.5740620 0.6220830 0.6927822    0
## 1000 0.4061697 0.5399732 0.5817349 0.5738875 0.6195585 0.6927822    0
## 2000 0.4061697 0.5308812 0.5917838 0.5768226 0.6220830 0.6927822    0
```

You have your final model. You can train the random forest with the following parameters:

- `ntree =800`: 800 trees will be trained
- `mtry=4`: 4 features is chosen for each iteration
- `maxnodes = 24`: Maximum 24 nodes in the terminal nodes (leaves)


```r
fit_rf <- train(survived ~., 
                data_train,
                method="rf",
                metric="Accuracy",
                tuneGrid=tuneGrid,
                trControl=trControl, 
                importance = TRUE,
                nodesize = 14,
                ntree =800,
                maxnodes=24)
```

**Step 5:  Evaluate the model**

The library `caret` has a function to make prediction. 

```
predict(model, newdata= df)
argument

- `model`: Define the model evaluated before. 
- `newdata`: Define the dataset to make prediction
```


```r
prediction <-predict(fit_rf, data_test)
```

You can use the prediction to compute the confusion matrix and see the accuracy score


```r
confusionMatrix(prediction, data_test$survived)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  No Yes
##        No  109  32
##        Yes  12  56
##                                           
##                Accuracy : 0.7895          
##                  95% CI : (0.7279, 0.8427)
##     No Information Rate : 0.5789          
##     P-Value [Acc > NIR] : 1.103e-10       
##                                           
##                   Kappa : 0.5544          
##  Mcnemar's Test P-Value : 0.004179        
##                                           
##             Sensitivity : 0.9008          
##             Specificity : 0.6364          
##          Pos Pred Value : 0.7730          
##          Neg Pred Value : 0.8235          
##              Prevalence : 0.5789          
##          Detection Rate : 0.5215          
##    Detection Prevalence : 0.6746          
##       Balanced Accuracy : 0.7686          
##                                           
##        'Positive' Class : No              
## 
```

You have an accuracy of 0.7943 percent, which is higher than the default value. 

Lastly, you can look at the feature importance with the function  `varImp()`. It seems that the most important features are the `sex` and `age`. That is not surprising because the important features are likely to appear closer to the root of the tree, while les  important features will often appear closed to the leaves.


```r
varImp(fit_rf)
```

```
## rf variable importance
## 
##              Importance
## sexmale         100.000
## fare             27.347
## age              27.029
## pclassUpper      19.965
## pclassMiddle     18.875
## parch             8.876
## sibsp             8.195
## embarkedC         3.539
## embarkedQ         1.142
## embarkedS         0.000
```

### Summary

We can summarize how to train and evaluate a random forest with the table below:

| Library      | Objective                        | function          | parameter                                                                                   |
|--------------|----------------------------------|-------------------|---------------------------------------------------------------------------------------------|
| randomForest | Create a Random forest           | RandomForest()    | formula, ntree=n, mtry=FALSE, maxnodes = NULL                                               |
| caret        | Create K folder cross validation | trainControl()    | method = "cv", number = n, search ="grid"                                                   |
| caret        | Train a Random Forest            | train()           | formula, df, method = "rf", metric= "Accuracy", trControl = trainControl(), tuneGrid = NULL |
| caret        | Predict out of sample            | predict           | model, newdata= df                                                                          |
| caret        | Confusion Matrix and Statistics  | confusionMatrix() | model, y test                                                                               |
| caret        | variable importance              | cvarImp()         | model                                                                                       |

### Appendix

List of model used in `caret`. Only the first tenth are returned.


```r
names(getModelInfo())[1:10]
```

```
##  [1] "ada"         "AdaBag"      "AdaBoost.M1" "adaboost"    "amdai"      
##  [6] "ANFIS"       "avNNet"      "awnb"        "awtan"       "bag"
```



