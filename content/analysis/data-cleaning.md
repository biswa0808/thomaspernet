---
title: Data Cleaning
author: Thomas
date: []
slug: data-cleaning
categories: []
tags:
  - preparation
header:
  caption: ''
  image: ''
---

## Data cleaning

Missing values in data science arise when an observation is missing in a column of a data frame or contains a character value instead of numeric value. Missing values must be dropped or replaced in order to draw correct conclusion from the data.

In this chapter, you will learn how to deal with missing values with the `dplyr` library. `dplyr` library is part of an ecosystem to realize a data analysis. 

**Function `mutate()`**

The fourth verb in the `dplyr` library is helpful to create new variables or change the values of existing variables. 

You will proceed in two parts. You will learn how to:

- exclude missing values from a data frame
- impute missing values with the mean and median

The verb `mutate()` is very easy to use. You can create a new variable following this syntax:

```
mutate(df, name_variable_1 = condition, ...)
arguments

- df: Data frame used to create a new variable
- name_variable_1: Name and the formula to create the new variable
- ...: No limit constraint. Possibility to create more than one variable inside `mutate()
```

### Exclude missing values

The `na.omit()` method from the `dplyr` library is a simple way to exclude missing observation. Dropping all the `NA` from the data is easy but it does not mean it is the most elegant solution. During analysis, it is wise to use variety of methods to deal with missing values

To tackle the problem of missing observations, you will use the titanic dataset. In this dataset, you have access to the information of the passengers on board during the tragedy. This dataset has many `NA` that need to be taken care of.

You will upload the csv file from the internet and then check which columns have `NA. To return the columns with missing data, you can use the following code:

Let's upload the data and verify the missing values.


```r
PATH <- "https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/titanic_csv.csv"
df_titanic <- read.csv(PATH, sep = ",")
# Return the column names containing missing observations
list_na <- colnames(df_titanic)[ apply(df_titanic, 2, anyNA) ]
list_na
```

```
## [1] "age"  "fare"
```

Here, 

```
colnames(df_titanic)[ apply(df_titanic, 2, anyNA) ]
```
Gives the name of columns that do not have data.

The columns `age` and `fare` have missing values. 

You can drop them with the `na.omit()`. 


```r
library(dplyr)
# Exclude the missing observations
df_titanic_drop <- df_titanic %>%
                   na.omit()
dim(df_titanic_drop)
```

```
## [1] 1045   13
```
The dataset contains 1045 rows compared to 1309 with the original dataset. 

### Impute missing values

You could also impute (populate) missing values with the median or the mean. A good practice is to create two separate variables for the mean and the median. Once created, you can replace the missing values with the newly formed variables. 

You will use the apply method to compute the mean of the column with `NA`. Let's see an example

**Step 1**

Earlier in the tutorial, you stored the columns name with the missing values in the list called list_na. You will use this list

**Step 2**

Now you need to compute of the mean with the argument `na.rm = TRUE. This argument is compulsory because the columns have missing values, and this tells R to ignore them.



```r
# Create mean
average_missing <- apply(df_titanic[,colnames(df_titanic) %in% list_na],
      2,
      mean,
      na.rm =  TRUE)
average_missing
```

```
##      age     fare 
## 29.88113 33.29548
```

Code Explanation: 

You pass 4 arguments in the apply method. 

-	df: `df_titanic[,colnames(df_titanic) %in% list_na]`. This code will return the columns name from the list_na object (i.e. "age" and "fare")
		`2: Compute the function on the columns
		`mean`: Compute the mean
		`na.rm = TRUE`: Ignore the missing values


You successfully created the mean of the columns containing missing observations. These two values will be used to replace the missing observations. 

**Step 3**

Replace the `NA` Values

The verb `mutate()` from the `dplyr` library is useful in creating a new variable. You don't necessarily want to change the original column so you can create a new variable without the `NA`. Mutate is easy to use, you just choose a variable name and define how to create this variable. 

Here is the complete code


```r
# Create a new variable with the mean and median
df_titanic_replace <- df_titanic %>%
            mutate(replace_mean_age  = ifelse(is.na(age), average_missing[1], age), 
                   replace_mean_fare = ifelse(is.na(fare), average_missing[2], fare))

sum(is.na(df_titanic_replace$age))
```

```
## [1] 263
```

```r
sum(is.na(df_titanic_replace$replace_mean_age))
```

```
## [1] 0
```

Code Explanation:

You create two variables, replace_mean_age and replace_mean_fare as follow:

-	`replace_mean_age  = ifelse(is.na(age), average_missing[1], age)`
		`replace_mean_fare = ifelse(is.na(fare), average_missing[2], fare)

If the column `age` has missing values, then replace with the first element of `average_missing` (mean of age), else keep the original values. Same logic for `fare`.


```r
sum(is.na(df_titanic_replace$age))
```

```
## [1] 263
```

```r
## [1] 263
sum(is.na(df_titanic_replace$replace_mean_age))
```

```
## [1] 0
```

```r
## [1] 0
```

The original column age has 263 missing values while the newly created variable have replaced them with the mean of the variable `age`. 

**Step 4**

You can replace the missing observation with the median as well. 


```r
median_missing <- apply(df_titanic[,colnames(df_titanic) %in% list_na],
      2,
      median,
      na.rm =  TRUE)

df_titanic_replace <- df_titanic %>%
            mutate(replace_median_age  = ifelse(is.na(age), median_missing[1], age), 
                   replace_median_fare = ifelse(is.na(fare), median_missing[2], fare))
head(df_titanic_replace)
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
##                         home.dest replace_median_age replace_median_fare
## 1                    St Louis, MO            29.0000            211.3375
## 2 Montreal, PQ / Chesterville, ON             0.9167            151.5500
## 3 Montreal, PQ / Chesterville, ON             2.0000            151.5500
## 4 Montreal, PQ / Chesterville, ON            30.0000            151.5500
## 5 Montreal, PQ / Chesterville, ON            25.0000            151.5500
## 6                    New York, NY            48.0000             26.5500
```

**Step 5**

A big data set could have lots of missing values and the above method could be cumbersome. You can execute all the above steps above in one line of code using `sapply() method. Though you would not know the vales of mean and median. 

`sapply` does not create a data frame, so you can wrap the `sapply()` function within `data.frame()` to create a data frame object.


```r
# Quick code to replace missing values with the mean
df_titanic_impute_mean <- data.frame(
  sapply(
    df_titanic,
    function(x) ifelse(is.na(x),
                       mean(x, na.rm = TRUE),
                       x)))
```

### Summary

You have three methods to deal with missing values:

-	Exclude all of the missing observations
		Impute with the mean 
		Impute with the median

The table below summarizes how to remove all the missing observations

| Library | Objective                 | Code                              |
|---------|---------------------------|-----------------------------------|
| base    | List missing observations | colnames(df)[apply(df, 2, anyNA)] |
| dplyr   | Remove all missing values | na.omit(df)                       |

The table below summarises the two ways to impute missing values.

| Method       | Objective      | Details                                                                                           | Advantages                     | Disavantages                            |
|--------------|----------------|---------------------------------------------------------------------------------------------------|--------------------------------|-----------------------------------------|
| Step by step | Impute missing | Check columns with missing, compute mean/median, store the value, replace with mutate()           | Know the value of means/median | More step. Can be slow with big dataset |
| Quick way    | Impute missing | Use sapply() and data.frame() to automatically search and replace missing values with mean/median | Short code and fast            | Don't know the imputation values        |

Example of the codes with the mean:

- Step by step


```r
# Step 1: Check columns with missing
list_na <- colnames(df)[ apply(df, 2, anyNA) ]
# Step 2: Compute mean/median
average_missing <- apply(df[,colnames(df) %in% list_na],
      2,
      mean,
      na.rm =  TRUE)
# Step 3: Replace with mutate() 
df_mean_replace <- df %>%
            mutate(replace_mean_X1  = ifelse(is.na(X1), average_missing[1], X1))
```

- Quick way


```r
data.frame(sapply(df,function(x) ifelse(is.na(x),mean(x, na.rm = TRUE),x)))
```

