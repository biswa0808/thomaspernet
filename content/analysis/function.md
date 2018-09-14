---
title: Function
author: Thomas
date: []
slug: function
categories: []
tags:
  - program
header:
  caption: ''
  image: ''
---

# Introduction to programming 

## Function in R

A **function**, in a programming environment, is a set of instructions. A programmer builds a function to avoid **repeating** the same task, or reduce **complexity**.

A function should be 

-	written to carry out a specified a tasks
-	may or may not include arguments
-	contain a body 
-	may or may not return one or more values.

In this chapter, we will learn how to use built-in function and how to create our own function. The roadmap is:

- R important built-in function
- Write function in R
- Environment scoping

A general approach of function is to use the argument part as **inputs**, feed the **body** part and finaly return an **output**. The Syntax of a function is the following:

```
function (arglist)  {
  #Function body
}
```

### R important buil-in functions

There are a lot of built-in function in R. R matches your input parameters with its function arguments, either by value or by position, then executes the function body. Function arguments can have default values: if you do not specify these arguments, R will take the default value.


```r
sort
```

```
## function (x, decreasing = FALSE, ...) 
## {
##     if (!is.logical(decreasing) || length(decreasing) != 1L) 
##         stop("'decreasing' must be a length-1 logical vector.\nDid you intend to set 'partial'?")
##     UseMethod("sort")
## }
## <bytecode: 0x7faca8937688>
## <environment: namespace:base>
```

*note*: It is possible to see the source code of a function by running the name of the function itself in the console. 

We will see three groups of function in action

- General function
- Math's function
- Statistical function

### General functions

We are already familiar with `cbind()`, `rbind()`, `range()`, `sort()`, `order()` functions. Each of these functions has a specific task, takes arguments to return an output. 

Following are important functions one must know:

**the function `diff()`**

If you work on **time serie**s, you need to stationary the series by taking their lag values. A **stationary process** allows constant mean, variance and autocorrelation over time. This mainly improves the prediction of a time series. It can be easily done with the function `diff()`. You can build a random time-series data with a trend and then use the function `diff()` to stationary the series. The `diff() function accepts one argument, a vector, and return suitable lagged and iterated difference.

**the function `length()`**

*note*: We often need to create random data, but for learning and comparison we want the numbers to be identical across machines. To ensure we all generate the same data, we use the `set.seed()` function with arbitrary values of 123.  The `set.seed()` function is generated through the process of pseudorandom number generator that make every modern computers to have the same sequence of numbers. If we don't use set.seed() function, we will all have different sequence of numbers.


```r
set.seed(123)
## Create the data
x = rnorm(1000)
ts <- cumsum(x)
## Stationary the serie
diff_ts <- diff(ts)
par(mfrow=c(1,2))
## Plot the series
plot(ts, type='l')
plot(diff(ts), type='l')
```

![](04-part_2_files/figure-epub3/ts-1.png)<!-- -->

In many cases, you want to know the **length** of a vector to make computation or in `for` loop. The `length()` function counts the number of rows in vector x.  The following codes import the `cars` dataset and return the number of rows. 

*note*:  `length()` returns the number of elements in a vector. If the function is passed into a matrix or a data frame, the number of columns is returned. 


```r
dt <- cars
## number columns
length(df)
```

```
## [1] 1
```

```r
## number rows
length(dt[,1])
```

```
## [1] 50
```

### Math functions

R integrates an array of mathematical functions. 

| Operator      | Description                                                                                  |
|---------------|----------------------------------------------------------------------------------------------|
| abs(x)        | Takes the absolute value of x                                                                |
| log(x,base=y) | Takes the logarithm of x with base y; if baseis not specified, returns the natural logarithm |
| exp(x)        | Returns the exponential of x                                                                 |
| sqrt(x)       | Returns the square root of x                                                                 |
| factorial(x)  | Returns the factorial of x (x!)                                                              |


```r
# Create a sequence of number from45 to 55
x_vector <- seq(45,55, by = 1)
#logarithm
log(x_vector)
```

```
##  [1] 3.806662 3.828641 3.850148 3.871201 3.891820 3.912023 3.931826
##  [8] 3.951244 3.970292 3.988984 4.007333
```

```r
#exponential
exp(x_vector)
```

```
##  [1] 3.493427e+19 9.496119e+19 2.581313e+20 7.016736e+20 1.907347e+21
##  [6] 5.184706e+21 1.409349e+22 3.831008e+22 1.041376e+23 2.830753e+23
## [11] 7.694785e+23
```

```r
#squared root
sqrt(x_vector)
```

```
##  [1] 6.708204 6.782330 6.855655 6.928203 7.000000 7.071068 7.141428
##  [8] 7.211103 7.280110 7.348469 7.416198
```

```r
#factorial
factorial(x_vector)
```

```
##  [1] 1.196222e+56 5.502622e+57 2.586232e+59 1.241392e+61 6.082819e+62
##  [6] 3.041409e+64 1.551119e+66 8.065818e+67 4.274883e+69 2.308437e+71
## [11] 1.269640e+73
```

### Statistical functions

R standard installation contains wide range of statistical functions. In this tutorial, we will briefly look at the most important function.

| Operator    | Description                                                                                      |
|-------------|--------------------------------------------------------------------------------------------------|
| mean(x)     | Mean of x                                                                  |
| median(x)   | Median of x                                                                |
| var(x)      | Variance of x              |
| sd(x)       | Standard deviation of x    |
| scale(x)    | Standard scores (z-scores) of x                                           |
| quantile(x) | The quartiles of x |
| sumary(x)   | Summary of x: mean, min, max etc..                                                     |


```r
speed <- dt$speed
# Mean speed of cars dataset
mean(speed)
```

```
## [1] 15.4
```

```r
# Median speed of cars dataset
median(speed)
```

```
## [1] 15
```

```r
# Variance speed of cars dataset
var(speed)
```

```
## [1] 27.95918
```

```r
# Standard deviation speed of cars dataset
sd(speed)
```

```
## [1] 5.287644
```

```r
# Standardise vector speed of cars dataset
head(scale(speed), 5)
```

```
##           [,1]
## [1,] -2.155969
## [2,] -2.155969
## [3,] -1.588609
## [4,] -1.588609
## [5,] -1.399489
```

```r
# Quantile speed of cars dataset
quantile(speed)
```

```
##   0%  25%  50%  75% 100% 
##    4   12   15   19   25
```

```r
# Summary speed of cars dataset
summary(speed)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##     4.0    12.0    15.0    15.4    19.0    25.0
```

Up to this point, we have learned a lot of R built-in functions. 

*note*: Be careful with the class of the argument, i.e. numeric, Boolean or string. For instance, if we need to pass a string value, we need to enclose the string in quotation mark: "ABC"" .

It is common practice to store the values of a function for future use. For instance, we can store the residuals of a simple linear model to check the normality hypothesis. 

### Write function in R

In some occasion, you need to write your own function because you have to accomplish a particular task and no function exists. A user-defined function involves a **name**, **arguments** and a **body**.

```
function.name <- function(arguments) 
{
  computations on the arguments
  some other code
}
```

You define the function similarly to variables, by **assigning** the directive **function(arguments)** to the **variable**, **function.name**, followed by the rest.

In the next session, we will go through step by step to create formulas with one or multiple arguments. 

*note*: A good practice is to name a user-defined function different from a built-in function. It avoids confusion. 

### One argument function

In the next snippet, you define a simple square function. The function passes a value and returns the square. 

```r
square_function<- function(n) 
{
  # compute the square of integer `n`
  n^2
}  
square_function(4)
```

```
## [1] 16
```

Code Explanation:

- The function is named square_function; it can be called whatever we want.
-	It receives an argument `n`. We didn't specify the type of variable so that the user can pass an integer, a vector or a matrix
-	The function takes the input `n` and returns the square of the input. 

When you are done using the function, you can remove it with the `rm()` function.
 

```r
rm(square_function)
square_function
```

On the console, we can see an error message :`Error: object 'square_function' not found` telling the function does not exist. 

### Environment scoping

In R, the **environment** is a **collection** of objects like functions, variables, data frame, etc.

R opens an environment each time Rstudio is prompted. 

The top-level environment available is the **global environment**, called `R_GlobalEnv`. And there is the **local environment**. 

You can list the content of the current environment.


```r
# Only the first 10th elements are listed
ls(environment())[1:10]
```

```
##  [1] "diff_ts"         "dt"              "speed"          
##  [4] "square_function" "ts"              "x"              
##  [7] "x_vector"        NA                NA               
## [10] NA
```

You can see all the variables and function created in the `R_GlobalEnv`. 

*note*: the argument `n` of the function `square_function` is **not in this global environment**. 

A **new** environment is created for each function. In the above example, the function `square_function()` creates a new environment inside the global environment.

To clarify the difference between **global** and **local environment**, you create two functions which return a similar output. These function takes a value `x` as an argument and add it to  `y` define *outside*  and *inside*  the function


```r
y <- 10
f <- function(x) {
  x + y
}

f(5)
```

```
## [1] 15
```

```r
y
```

```
## [1] 10
```

```r
rm(y)
```

The function  `f` returns the output 15; this is because `y` is defined in the global environment. Any variable defined in the global environment can be used locally. The variable `y` has the value of 10 during all scripts and can be accessible at any time.

Let's see what happens if the variable `y` is defined inside the function. 

We dropped `y` prior to run this code using `rm(r)`.


```r
f <- function(x) {
  y <- 10
  x + y
}
f(5)
y
```

The output is also 15 when you call `f(5)` but returns an error when you try to print the value `y`. The variable `y` is not in the global environment. 

Finally, R uses the most recent variable definition to pass inside the body of a function. Let's consider the following example:


```r
y <- 2
f <- function(x) {
  y <- 4
  x + y
}

f(5)
```

```
## [1] 9
```

R ignores the `y` values defined outside the function because we explicitly created a `y` variable inside the body of the function. 

### Many arguments function

You can write a function with more than one argument. Consider the function called `time`. It is a straightforward function multiplying two variables.


```r
times <- function(x,y) {
  x*y
}
times(2,4)
```

```
## [1] 8
```

### When shall we write function? 

Data scientist need to do many repetitive tasks. Most of the time, we copy and paste chunks of code repetitively. For example, normalization of  a variable is highly recommended before we run a machine learning algorithm. 

The formula to normalize a variable is:

$$
normalize= \frac{x-x_{min}}{x_{max}-x_{min}}
$$

You already know how to use the min() and max() function in R. You use the tibble library to create the data frame.
Tibble is so far the most convenient function to create a data set from scratch.


```r
library(tibble)
data_frame <- tibble(
  c1 = rnorm(50, 5, 1.5), 
  c2 = rnorm(50, 5, 1.5),
  c3 = rnorm(50, 5, 1.5)
)
```

You will proceed in two steps to compute the function described above. In the first step, you will create a variable called `c1_norm` which is the rescaling of `c1`. In step two, you just copy and paste the code of `c1_norm` and change with `c2` and `c3`.

Detail of the function with the column c1:

- Nominator: $x-x_{min}$ : `data_frame$c1 -min(data_frame$c1))`

- Denominator:$x_{max}-x_{min}$: `max(data_frame$c1)-min(data_frame$c1))`

Therefore, we can divide them to get the normalized value of column `c1`:

```
(data_frame$c1 -min(data_frame$c1))/(max(data_frame$c1)-min(data_frame$c1))
```

We can create `c1_norm`, `c2_norm` and `c3_norm`:


```r
# Create c1_norm: rescaling of c1
data_frame$c1_norm <- (data_frame$c1 -min(data_frame$c1))/(max(data_frame$c1)-min(data_frame$c1))
# show the first five values
head(data_frame$c1_norm, 5)
```

```
## [1] 0.2886787 0.2804736 0.4703763 0.4491567 0.0000000
```

It works. You can copy and paste 

```
data_frame$c1_norm <- (data_frame$c1 -min(data_frame$c1))/(max(data_frame$c1)-min(data_frame$c1))
```

then change `c1_norm` to `c2_norm` and `c1` to `c2`. We do the same to create `c3_norm`


```r
# Column c2_norm
data_frame$c2_norm <- (data_frame$c2 - min(data_frame$c2))/(max(data_frame$c2)-min(data_frame$c2))
# Column c3_norm
data_frame$c3_norm <- (data_frame$c3 - min(data_frame$c3))/(max(data_frame$c3)-min(data_frame$c3))
```

We perfectly rescaled the variables `c1`, `c2` and `c3`. 

However, this method is prone to mistake. We could copy and forget to change the column name after pasting. Therefore, a good practice is to write a function each time you need to paste same code more than twice. We can rearrange the code into a formula and call it whenever it is needed. To write our own function, we need to give:

-	 Name: `normalize`. 
-	the number of arguments: You only need one argument, which is the column we use in our computation. 
-	The body: This is simply the formula we want to return. 

We will proceed step by step to create the function normalize. 

**Step 1**

You create the nominator, which is $x-x_{min}$. In R, we can store the nominator in a variable like this:

```
nominator <- x-min(x)
```

**Step 2**

You compute the denominator: $x_{max}-x_{min}$. You can replicate the idea of step 1 and store the computation in a variable:

``` 
denominator <- max(x)-min(x)
```

**Step 3**

We perform the division between the nominator and denominator.

```
normalize <- nominator/denominator
```

**Step 4**

To return value to calling function you need to pass normalize inside `return() to get the output of the function.

```
return(normalize)
```

**Step 5**

You are ready to use the function by wrapping everything inside the bracket.


```r
normalize <- function(x){
  # step 1: create the nominator
  nominator <- x-min(x)
  # step 2: create the denominator
  denominator <- max(x)-min(x)
  # step 3: divide nominator by denominator
  normalize <- nominator/denominator
  # return the value
  return(normalize)
}
```

Let's test your function with the variable `c1`:


```r
normalize(data_frame$c1)
```

```
##  [1] 0.2886787 0.2804736 0.4703763 0.4491567 0.0000000 0.6670762 0.5201213
##  [8] 0.9226957 0.6010405 0.3906637 0.9935270 1.0000000 0.2472571 0.5608726
## [15] 0.4344636 0.5084751 0.5159993 0.2392318 0.5267855 0.7987615 0.4432263
## [22] 0.4434425 0.7335992 0.6406569 0.1673950 0.5161877 0.7809786 0.7367032
## [29] 0.5517526 0.6077342 0.2513036 0.5294877 0.2963623 0.3886087 0.6475707
## [36] 0.2624606 0.5233160 0.5533098 0.4839211 0.8123155 0.2837454 0.5863482
## [43] 0.4571924 0.4252497 0.5599546 0.2841072 0.2296527 0.3818332 0.7992279
## [50] 0.4840796
```

Functions are more comprehensive way to perform a repetitive task. You can use the normalize formula over different columns, like below:


```r
data_frame$c1_norm_function <- normalize(data_frame$c1)
data_frame$c2_norm_function <- normalize(data_frame$c2)
data_frame$c3_norm_function <- normalize(data_frame$c3)
```

Even though the example is simple, we can infer the power of a formula. The above code is easier to read and especially avoid to mistakes when pasting codes.

### Formulas with condition

Sometime, we need to include conditions into a formula to allow the code to return different outputs. 

In Machine Learning tasks, you need to split the dataset between a train set and a test set. The train set allows the algorithm to learn from the data. In order to test the performance of our model, you can use the test set to return the performance measure. R does not have a function to create two datasets. You can write our own function to do that. Our function takes two arguments and is called `split_data(). The idea behind is simple; you multiply the length of dataset (i.e. number of observations) with 0.8. For instance, if you want to split the dataset 80/20, and our dataset contains 100 rows, then our function will multiply 0.8*100 = 80. 80 rows will be selected to become our training data. 

You will use the `airquality` dataset to test our user-defined function. The airquality dataset has 153 rows. You can see it with the code below:


```r
nrow(airquality)
```

```
## [1] 153
```

We will proceed as follow:

```
split_data <- function(df, train = TRUE)
Arguments:

-	df: Define the dataset
-	train: Specify if the function returns the train set or test set. By default, set to TRUE
```

Your function has two arguments. The arguments train is a Boolean parameter. If it is set to `TRUE`, your function creates the train dataset, otherwise, it creates the test dataset. 

You can proceed like you did with the `normalise()` function. You write the code as if it was only one-time code and then wrap everything with the condition into the body to create the function. 

**Step 1**

You need to compute the length of the dataset. This is done with the function `nrow()`. `nrow()` returns the total number of rows in the dataset. You call the variable length. 


```r
lenght <- nrow(airquality)
length
```

```
## function (x)  .Primitive("length")
```

**Step 2**

You multiply the length by 0.8. It will return the number of rows to select. It should be 153*0.8 = 122.4


```r
total_row <- lenght *0.8
total_row
```

```
## [1] 122.4
```

You want to select 122 rows among the 153 rows in the airquality dataset. You create a list containing values from 1 to total_row. You store the result in the variable called split 


```r
split <- 1:total_row
split[1:5] 
```

```
## [1] 1 2 3 4 5
```

split chooses the first 122 rows from the dataset. For instance, you can see that your variable split gathers the value 1, 2, 3, 4, 5 and so on. These values will be the index when you will select the rows to return.

**Step 3**

We need to select the rows in the `airquality` dataset based on the values stored in the split variable. This is done like this:


```r
train_df <- airquality[split, ] 
head(train_df)
```

```
##   Ozone Solar.R Wind Temp Month Day
## 1    41     190  7.4   67     5   1
## 2    36     118  8.0   72     5   2
## 3    12     149 12.6   74     5   3
## 4    18     313 11.5   62     5   4
## 5    NA      NA 14.3   56     5   5
## 6    28      NA 14.9   66     5   6
```

**Step 4**

You can create the test dataset by using the remaining rows, 123:153. This is done by using `-` in front of split.


```r
test_df <- airquality[-split, ] 
head(test_df)
```

```
##     Ozone Solar.R Wind Temp Month Day
## 123    85     188  6.3   94     8  31
## 124    96     167  6.9   91     9   1
## 125    78     197  5.1   92     9   2
## 126    73     183  2.8   93     9   3
## 127    91     189  4.6   93     9   4
## 128    47      95  7.4   87     9   5
```

**Step 5**

You can create the condition inside the body of the function. Remember, you have an argument train that is a Boolean set to `TRUE` by default to return the train set. To create the condition, you use the if syntax: 

```
 if (train ==TRUE){ 
	train_df <- airquality[split, ] 
      return(train)
  } else {
	test_df <- airquality[-split, ] 
      return(test)
  }
```

This is it, you can write the function. You only need to change `airquality` to df because you want to try our function to any data frame, not only `airquality`:


```r
split_data <- function(df, train = TRUE){
  lenght <- nrow(df)
  split <- 1:total_row
  if (train ==TRUE){ 
	train_df <- df[split, ] 
      return(train_df)
  } else {
	test_df <- df[-split, ] 
      return(test_df)
  }
}
```

Let's try our function on the `aiquality` dataset. we should have one train set with 122 rows and a test set with 31 rows.

