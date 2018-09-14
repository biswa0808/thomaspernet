---
title: Decision Tree
author: Thomas
date: []
slug: decision-tree
categories: []
tags:
  - ml
header:
  caption: ''
  image: ''
---

## Decision trees

Decision trees are versatile Machine Learning algorithm that can perform both classification and regression tasks. They are very powerful algorithms, capable of fitting complex datasets. Besides, decision trees are fundamental components of random forests, which are among the most potent Machine Learning algorithms available today. 

In this session, you will do learn:

- Training and Visualizing a decision trees for a classification task
- Making prediction
- Regularization Hyper-parameters

### Training and Visualizing a decision trees

To build your first decision trees, we will proceed as follow:

- Step 1: Import the data
- Step 2: Clean the dataset
- Step 3: Create train/test set
- Step 4: Build the model
- Step 5: Make prediction
- Step 6: Measure performance
- Step 7: Tune the hyper-parameters

**Step 1: Import the data**

You load the titanic dataset. If you are curious about the fate of the titanic, you can watch this video on [Youtube](https://www.youtube.com/watch?v=9xoqXVjBEF8). The purpose of this dataset is to predict which people are more likely to survive after the collision with the iceberg. The dataset contains 13 variables and 1309 observations. The dataset is ordered by the variable `X`. 


```r
set.seed(678)
path <- 'https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/titanic_csv.csv'
titanic <- read.csv(path)
head(titanic)
```

```
##   X pclass survived                                            name    sex
## 1 1      1        1                   Allen, Miss. Elisabeth Walton female
## 2 2      1        1                  Allison, Master. Hudson Trevor   male
## 3 3      1        0                    Allison, Miss. Helen Loraine female
## 4 4      1        0            Allison, Mr. Hudson Joshua Creighton   male
## 5 5      1        0 Allison, Mrs. Hudson J C (Bessie Waldo Daniels) female
## 6 6      1        1                             Anderson, Mr. Harry   male
##       age sibsp parch ticket     fare   cabin embarked
## 1 29.0000     0     0  24160 211.3375      B5        S
## 2  0.9167     1     2 113781 151.5500 C22 C26        S
## 3  2.0000     1     2 113781 151.5500 C22 C26        S
## 4 30.0000     1     2 113781 151.5500 C22 C26        S
## 5 25.0000     1     2 113781 151.5500 C22 C26        S
## 6 48.0000     0     0  19952  26.5500     E12        S
##                         home.dest
## 1                    St Louis, MO
## 2 Montreal, PQ / Chesterville, ON
## 3 Montreal, PQ / Chesterville, ON
## 4 Montreal, PQ / Chesterville, ON
## 5 Montreal, PQ / Chesterville, ON
## 6                    New York, NY
```

```r
tail(titanic)
```

```
##         X pclass survived                      name    sex  age sibsp
## 1304 1304      3        0     Yousseff, Mr. Gerious   male   NA     0
## 1305 1305      3        0      Zabour, Miss. Hileni female 14.5     1
## 1306 1306      3        0     Zabour, Miss. Thamine female   NA     1
## 1307 1307      3        0 Zakarian, Mr. Mapriededer   male 26.5     0
## 1308 1308      3        0       Zakarian, Mr. Ortin   male 27.0     0
## 1309 1309      3        0        Zimmerman, Mr. Leo   male 29.0     0
##      parch ticket    fare cabin embarked home.dest
## 1304     0   2627 14.4583              C          
## 1305     0   2665 14.4542              C          
## 1306     0   2665 14.4542              C          
## 1307     0   2656  7.2250              C          
## 1308     0   2670  7.2250              C          
## 1309     0 315082  7.8750              S
```

From the head and tail output, you can notice the data is not shuffled. This is a big issue! When you will split your data between  a train set and test set, you will select **only** the passenger from class 1 and 2 (No passenger from class 3 are in the top 80 percent of the observations), which means the algorithm will never see the features of passenger of class 3. This mistake will lead to poor prediction. 

To overcome this issue, you can use the function `sample()`. 


```r
shuffle_index <- sample(1:nrow(titanic))
head(shuffle_index)
```

```
## [1]  288  874 1078  633  887  992
```

Code Explanation

- `sample(1:nrow(titanic))`: Generate a random list of index from 1 to 1309 (i.e. the maximum number of rows). 

You will use this index to shuffle the titanic dataset.


```r
titanic <- titanic[shuffle_index, ]
head(titanic)
```

```
##         X pclass survived
## 288   288      1        0
## 874   874      3        0
## 1078 1078      3        1
## 633   633      3        0
## 887   887      3        1
## 992   992      3        1
##                                                           name    sex age
## 288                                      Sutton, Mr. Frederick   male  61
## 874                   Humblen, Mr. Adolf Mathias Nicolai Olsen   male  42
## 1078                                 O'Driscoll, Miss. Bridget female  NA
## 633  Andersson, Mrs. Anders Johan (Alfrida Konstantia Brogren) female  39
## 887                                        Jermyn, Miss. Annie female  NA
## 992                                           Mamee, Mr. Hanna   male  NA
##      sibsp parch ticket    fare cabin embarked           home.dest
## 288      0     0  36963 32.3208   D50        S     Haddenfield, NJ
## 874      0     0 348121  7.6500 F G63        S                    
## 1078     0     0  14311  7.7500              Q                    
## 633      1     5 347082 31.2750              S Sweden Winnipeg, MN
## 887      0     0  14313  7.7500              Q                    
## 992      0     0   2677  7.2292              C
```

**Step 2: Clean the data**

The structure of the data shows some variables have `NA`'s. Data clean up to be done as follows:

- Drop variables `home.dest`,`cabin`, `name`, `X` and `ticket`
- Create factor variables for `pclass` and `survived`
- Drop the `NA`


```r
library(dplyr)
# Drop variables
clean_titanic <- titanic %>%
                 select(-c(home.dest, cabin, name, X, ticket)) %>%
# Convert to factor level  
                 mutate(pclass = factor(pclass, levels = c(1,2,3), labels= c('Upper', 'Middle', 'Lower')),
                        survived = factor(survived, levels = c(0,1), labels = c('Died', 'Survived'))) %>%
  na.omit()

glimpse(clean_titanic)
```

```
## Observations: 1,045
## Variables: 8
## $ pclass   <fct> Upper, Lower, Lower, Upper, Middle, Upper, Middle, Up...
## $ survived <fct> Died, Died, Died, Survived, Died, Survived, Survived,...
## $ sex      <fct> male, male, female, female, male, male, female, male,...
## $ age      <dbl> 61.0, 42.0, 39.0, 49.0, 29.0, 37.0, 20.0, 54.0, 2.0, ...
## $ sibsp    <int> 0, 0, 1, 0, 0, 1, 0, 0, 4, 0, 0, 1, 1, 0, 0, 0, 1, 1,...
## $ parch    <int> 0, 0, 5, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 2, 0, 4, 0,...
## $ fare     <dbl> 32.3208, 7.6500, 31.2750, 25.9292, 10.5000, 52.5542, ...
## $ embarked <fct> S, S, S, S, S, S, S, S, S, C, S, S, S, Q, C, S, S, C,...
```

Before you train your model, you need to add two steps:

- Create a train and test set: You train the model on the train set and test the prediction on the test set (i.e. unseen data)
- Install `rpart.plot` from the console

**Step 3: Create train/test set**

The common practice is to split the data 80/20, 80 percent of the data serves to train the model, and 20 percent to make predictions. You need to create two separate data frames. You don't want to touch the test set until you finish to build your model. You can create a function name `create_train_test()` that takes three arguments.

```
create_train_test(df, size = 0.8, train = TRUE)
arguments:

- df: Dataset used to train the model.
- size: Size of the split. By default, 0.8. Numerical value
- train: If set to `TRUE`, the function creates the train set, otherwise the test set. Default value sets to `TRUE`. Boolean value.
```

You need to add a Boolean parameter because R does not allow to return two data frames simultaneously.


```r
create_train_test <- function(data, size=0.8, train = TRUE){
  n_row = nrow(data)
  total_row = size*n_row
  train_sample <- 1:total_row
  if (train ==TRUE){ 
    return(data[train_sample, ])
  } else {
    return(data[-train_sample, ])
  }
}
```

Code Explanation

- `function(data, size=0.8, train = TRUE)`: Add the arguments in the function
- `n_row = nrow(data)`: Count number of rows in the dataset
- `total_row = size*n_row`: Return the nth row to construct the train set
- `train_sample <- 1:total_row`: Select the first row to the nth rows
- `if (train ==TRUE){ } else { }`: If condition sets to true, return the train set, else the test set.

You can test your function and check the dimension.


```r
data_train <-create_train_test(clean_titanic, 0.8, train = TRUE)
data_test <-create_train_test(clean_titanic, 0.8, train = FALSE)

dim(data_train)
```

```
## [1] 836   8
```

```r
dim(data_test)
```

```
## [1] 209   8
```

The train dataset has 1046 rows while the test dataset has 262 rows. 

You use the function `prop.table()` combined with `table()` to verify if the randomization process is correct. 


```r
prop.table(table(data_train$survived))
```

```
## 
##      Died  Survived 
## 0.5944976 0.4055024
```

```r
prop.table(table(data_test$survived))
```

```
## 
##      Died  Survived 
## 0.5789474 0.4210526
```

In both dataset, the amount of survivors is the same, about 40 percent. 

**Install `rpart.plot`**

`rpart.plot` is not available from `conda` libraries. You can install it from the console.



```r
install.packages("rpart.plot")
```

**Step 4: Build the model**

You are ready to build the model. The syntax for `Rpart()` function is:

```
rpart(formula, data=, method='')
arguments:
  
- formula: The function to predict
- data: Specifies the data frame
- method: 
    - "class" for a classification tree 
    - "anova" for a regression tree
```

You use the `class` method because you predict a class.


```r
library(rpart)
library(rpart.plot)
fit <- rpart(survived ~., data = data_train, method = 'class')
```

Code Explanation

- `rpart()`: Function to fit the model. The arguments are:
    - `survived ~.`: Formula of the Decision Trees
    - `data = data_train`: Dataset
    - `method = 'class'`:  Fit a binary model
    
You can plot the model


```r
rpart.plot(fit, extra= 101)
```

![](/project/decision-tree_files/43.png)


Code Explanation

- `rpart.plot(fit, extra= 104)`: Plot the tree. The `extra` features are set to `106` to display the probability of the 2nd class (useful for binary responses). You can refer to the [vignette](https://cran.r-project.org/web/packages/rpart.plot/rpart.plot.pdf) for more information about the other choices.      

You start at the *root node* (depth 0 over 3, the top of the graph): 

1. At the top, it is the overall probability of survival. It shows the proportion of passenger that survided to the crash. 41 percent of passenger survived. 
2. This node asks whether the gender of the passenger is male. If yes, then you go down to the root's left child node (depth 2). 63 percent are males with a survival probability of 21 percent.
3. In the second node, you ask if the male passenger is above 3.5 years old. If yes, then the chance of survival is 19 percent. 
4. You keep on going like that to understand what features impact the likelihood of survival.  

*note*: one of the many qualities of Decision Trees is that they require very little data preparation. In particular, they don't require feature scaling or centering. 

By default, `rpart()` function uses the **Gini** impurity measure to split the note. The higher the Gini coefficient, the more different instances within the node. 

**Step 5: Make prediction**

You can predict your test dataset. To make a prediction, you can use the `predict()` function. The basic syntax of predict for a decision trees is:

```
predict(fitted_model, df, type = 'class')
arguments:

- fitted_model: This is the object stored after a model estimation. 
- df: Data frame used to make the prediction
- type: Type of prediction
    - 'class': for classification
    - 'prob': to compute the probability of each class
    - 'vector': Predict the mean response at the node level
```

You want to predict which passengers are more likely to survive after the collision from the test set. It means, you will know among those 209 passengers, which one will survive or not.


```r
predict_unseen <- predict(fit, data_test, type = 'class')
```

Code Explanation

- `predict(fit, data_test, type = 'class')`:  Predict the class (0/1) of the test set

You actually know the passenger that didn't make it and those who did. 


```r
table_mat <- table(data_test$survived, predict_unseen)
table_mat
```

```
##           predict_unseen
##            Died Survived
##   Died      106       15
##   Survived   30       58
```

Code Explanation

- `table(data_test$survived, predict_unseen)`: Create a $2*2$ table to count how many passengers are classified as survivors and passed away compare to the correct classification

The model correctly predicted 106 dead passengers but classified 15 survivors as dead. By analogy, the model misclassified 30 passengers as survivors while they turned out to be dead.

**Step 6: Measure performance**

You can compute an accuracy measure for classification task with the **confusion matrix**:

The **confusion matrix** is a better choice to evaluate the classification performance. The general idea is to count the number of times `True` instances are classified are `False`. 


![](/project/decision-tree_files/44.png)

Each row in a confusion matrix represents an *actual target*, while each column represents a *predicted target*. The first row of this matrix considers dead passengers (the False class): 106 were correctly classified as dead (**True negative**), while the remaining one was wrongly classified as survivor (**False positive**). The second row considers the survivors, the positive class were 58 (**True positive**), while the **True negative** was 30. 

You can compute the **accuracy test** from the confusion matrix: 

$$
accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$

This is the proportion of true positive and true negative over the sum of the matrix.  With R, you can code as follow:


```r
accuracy_Test <- sum(diag(table_mat))/sum(table_mat)
```

Code Explanation

- `sum(diag(table_mat))`: Sum of the diagonal 
- `sum(table_mat)`: Sum of the matrix.

You can print the accuracy of the test set:


```r
print(paste('Accuracy for test', accuracy_Test))
```

```
## [1] "Accuracy for test 0.784688995215311"
```

You have a score of 78 percent for the test set 
You can replicate the same exercise with the train dataset. 

**Step 7: Tune the hyper-parameters**

Decision tree has various parameters that control aspects of the fit. In `rpart` library, you can control the parameters using the `rpart.control()` function. In the following code, you introduce the parameters you will tune. You can refer to the [vignette](https://cran.r-project.org/web/packages/rpart/rpart.pdf) for other parameters.

```
rpart.control(minsplit = 20, minbucket = round(minsplit/3), maxdepth = 30)
Arguments:
  
- minsplit: Set the minimum number of observations in the node before the algorithm perform a split
- minbucket:  Set the minimum number of observations in the final note i.e. the leaf
- maxdepth: Set the maximum depth of any node of the final tree. The root node is treated a depth 0
```

We will proceed as follow:

- Construct function to return accuracy
- Tune the maximum depth
- Tune the minimum number of sample a node must have before it can split
- Tune the minimum number of sample a leaf node must have

You can write a function to display the accuracy. You simply wrap the code you used before:

1. predict: `predict_unseen <- predict(fit, data_test, type = 'class')`
2. Produce table: `table_mat <- table(data_test$survived, predict_unseen)`
3. Compute accuracy: `accuracy_Test <- sum(diag(table_mat))/sum(table_mat)`


```r
accuracy_tune <- function(fit){
  predict_unseen <- predict(fit, data_test, type = 'class')
  table_mat <- table(data_test$survived, predict_unseen)
  accuracy_Test <- sum(diag(table_mat))/sum(table_mat)
  accuracy_Test
}
```

You can try to tune the parameters and see if you can improve the model over the default value. As a reminder, you need to get an accuracy higher than 0.78

You run the algorithm with the following parameters: 

```
minsplit = 4
minbucket= round(5/3)
maxdepth = 3
cp=0
```


```r
control <- rpart.control(minsplit = 4,
                         minbucket= round(5/3),
                         maxdepth = 3,
                         cp=0)
tune_fit <- rpart(survived ~., data = data_train, method = 'class', control = control)
accuracy_tune(tune_fit)
```

```
## [1] 0.7990431
```

You get a higher performance than the previous model. Congratulation! 

### Summary

We can summarize the functions to train a decision trees algorithm.

| Library | Objective                          | function        | class  | parameters                   | details                                                                                 |
|---------|------------------------------------|-----------------|--------|------------------------------|-----------------------------------------------------------------------------------------|
| rpart   | Train classification trees         | rpart()         | class  | formula, df, method          |                                                                                         |
| rpart   | Train regression tree              | rpart()         | anova  | formula, df, method          |                                                                                         |
| rpart   | Plot the trees                     | rpart.plot()    |        | fitted model                 |                                                                                         |
| base    | predict                            | predict()       | class  | fitted model, type           |                                                                                         |
| base    | predict                            | predict()       | prob   | fitted model, type           |                                                                                         |
| base    | predict                            | predict()       | vector | fitted model, type           |                                                                                         |
| rpart   | Control parameters                 | rpart.control() |        | minsplit                     | Set the minimum number of observations in the node before the algorithm perform a split |
|         |                                    |                 |        |                              |                                                                                         |
|         |                                    |                 |        | minbucket                    | Set the minimum number of observations in the final note i.e. the leaf                  |
|         |                                    |                 |        |                              |                                                                                         |
|         |                                    |                 |        | maxdepth                     | Set the maximum depth of any node of the final tree. The root node is treated a depth 0 |
| rpart   | Train model with control parameter | rpart()         |        | formula, df, method, control |                                                                                         |

*note*: Train the model on a training data and test the performance on an unseen dataset, i.e. test set. 

