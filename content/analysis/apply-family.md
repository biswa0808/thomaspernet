---
title: Apply Family
author: Thomas
date: []
slug: apply-family
categories: []
tags:
  - program
header:
  caption: ''
  image: ''
---

## Apply Family

This chapter aims at introducing the `apply()` function collection. The `apply()` function is the most basic of the collection, we will talk as well about `sapply()`, `lapply()` and `tapply()`. The apply collection can be viewed as a substitute to the loop

The `apply()` collection comes from with **r essential** if we install R with Anaconda. The `apply()` function can be feed with many functions to perfom redundant application on a collection of object (data frame, list, vector, etc.). The purpose of `apply()` is primarily to avoid explicit uses of loop constructs. They can be used for an input list, matrix or array and apply a function. Any function can be passed into `apply()`.

### What occasions to use `apply()` collection

We use `apply()` over a matrice. This function takes 5 arguments:

```
apply(X, MARGIN, FUN)
arguments

- x: an array or matrix
- MARGIN:  take a value or range between 1 and 2 in order to define where to apply the function:
     - `MARGIN=1`: the manipulation is performed on rows
     - `MARGIN=2`: the manipulation is performed on columns
     - `MARGIN=c(1,2)` the manipulation is performed on rows and columns
- FUN: tells which function to apply  
```

The simplest example is to sum a matrice over all the columns. The code `apply(m1, 2, sum)` will apply the sum function to the matrix 5x6 and return the sum of each column accessible in the dataset. 


```r
m1 <- matrix(rnorm(30), nrow=5, ncol=6)
a_m1 <- apply(m1, 2, sum)
a_m1
```

```
## [1]  1.2370231 -3.6465888 -1.9764360 -0.3839391 -2.4636865  0.7043155
```

A best practice is to store the values before printing it to the console.

As a reminder, R has different build-in functions:

- `mean(x, na.rm = FALSE)`
- `sd(x)`
- `median(x)`
- `sum(x)`
- `min(x)`
- `max(x)`
- `scale(x)`
- `abs(x)`
- `sqrt(x)`
- `round(x, digit = n)`
- `log(x)`
- `exp(x)
- ...

The syntax of `lapply()` is:

```
lapply(X, FUN)
Arguments:
  
- X: A vector or an object
- FUN: Function applied to each element of x
```

`l` in `lapply()` stands for list. The difference between `lapply()` and `apply()` lies between the output return. The output of `lapply()` is a list. `lapply()` can be used for other objects like data frames  and lists.

`lapply()` function does not need `MARGIN`. 

A very easy example can be to change the string value of a matrix to lower case with `tolower` function. We construct a matrix with the name of the famous movies. The name is in upper case format.


```r
movies <- c("SPYDERMAN", "BATMAN", "VERTIGO", "CHINATOWN")
movies_lower <-lapply(movies, tolower)
str(movies_lower)
```

```
## List of 4
##  $ : chr "spyderman"
##  $ : chr "batman"
##  $ : chr "vertigo"
##  $ : chr "chinatown"
```

We can use `unlist()` to convert the list into a vector. 


```r
movies_lower <-unlist(lapply(movies, tolower))
str(movies_lower)
```

```
##  chr [1:4] "spyderman" "batman" "vertigo" "chinatown"
```

`sapply()` function does the same jobs as `lapply()` function but returns a vector.

```
sapply(X, FUN)
Arguments:
  
- X: A vector or an object
- FUN: Function applied to each element of x
```

We can measure the minimum speed and stopping distances of cars from the `cars` dataset. 


```r
dt <- cars
# Minimum
lmn_cars <- lapply(dt, min)
smn_cars <- sapply(dt, min)
lmn_cars
```

```
## $speed
## [1] 4
## 
## $dist
## [1] 2
```

```r
smn_cars
```

```
## speed  dist 
##     4     2
```

```r
# maximum
lmxcars <- lapply(dt, max)
smxcars <- sapply(dt, max)
lmxcars
```

```
## $speed
## [1] 25
## 
## $dist
## [1] 120
```

```r
smxcars
```

```
## speed  dist 
##    25   120
```

We can use a user built-in function into `lapply()` or `sapply()`. We create a function named `avg` to compute the average of the minimum and maximum of the vector. 


```r
avg <- function(x) {
  ( min(x) + max(x) ) / 2
}

fcars <- sapply(dt, avg)
fcars
```

```
## speed  dist 
##  14.5  61.0
```

`sapply()` function is more efficient than `lapply()` in the output returned because `sapply()` store values direclty into a vector. In the next example, you will see this is not always the case.

We can summarize the difference between `apply()`, `sapply()` and `lapply() in the following table:

| Function | Arguments             | Objective                                         | Input                      | Output              |
|----------|-----------------------|---------------------------------------------------|----------------------------|---------------------|
| apply    | apply(x, MARGIN, FUN) | Apply a function to the rows or columns or both   | Data frame or matrix       | vector, list, array |
| lapply   | lapply(X, FUN)        | Apply a function to all the elements of the input | List, vector or data frame | list                |
| sapply   | sappy(X FUN)          | Apply a function to all the elements of the input | List, vector or data frame | vector or matrix    |

### Slice vector

You can use `lapply()` or `sapply()` interchangeable to slice a data frame. We create a function, `below_average()`, that takes a vector of numerical values and returns a vector that only contains the values that are strictly above the average. You compare both results with the `identical()` function.


```r
below_ave <- function(x) {
  ave <- mean(x)
  return(x[x > ave])
}

dt_s<- sapply(dt, below_ave)
dt_l<- lapply(dt, below_ave)
identical(dt_s, dt_l)
```

```
## [1] TRUE
```

The function `tapply()` computes a measure (mean, median, min, max, etc..) or a function for each factor variable in a vector. 

```
tapply(X, INDEX, FUN = NULL)
Arguments:
  
- X: An object, usually a vector
- INDEX: A list containing factor
- FUN: Function applied to each element of x
```

Part of the job of a data scientist or researchers is to compute summaries of variables. For instance, measure the average or more complex function with eigen values. Most of the data are grouped by ID, city, countries, and so on. Summarizing over group reveals more interesting patterns. 

To understand how it works, let's use the `iris` dataset. This dataset is very famous in the world of machine learning. The purpose of this dataset is to predict the class of each of the three flower species:  `Sepal`, `Versicolor`, `Virginica`. The dataset collects information for each species about their length  and width. 

As a prior work, you can compute the median of the length for each species. `tapply()` is a quick way to perform this computation.  


```r
data(iris)
tapply(iris$Sepal.Width, iris$Species, median)
```

```
##     setosa versicolor  virginica 
##        3.4        2.8        3.0
```
