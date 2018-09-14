---
title: Data Types
author: Thomas
date: '2018-08-11'
slug: data-types
categories: []
tags:
  - intro
header:
  caption: ''
  image: ''
---

# Introduction of R framework

The first part of the book introduces you the different types of data and how to manipulate them in the R environment.

## Data type

### Basic operations

We will first see the basic arithmetic operations in R. The following operators stand for:

| Operator | Description    |
|----------|----------------|
| +        | Addition       |
| -        | Substraction   |
| *        | Multiplication |
| /        | Division       |
| ^ or **  | Exponentiation |


```r
# An addition
3 + 4
```

```
## [1] 7
```

```r
# A multiplication
3 * 5
```

```
## [1] 15
```

```r
 # A division
(5 + 5) / 2 
```

```
## [1] 5
```

```r
# Exponentiation
2^5
```

```
## [1] 32
```

```r
# Modulo
28 %% 6
```

```
## [1] 4
```

### Variables

Variables store values and are an important component in programming, especially for a data scientist. A variable can store a number, an object, a statistical result, vector, dataset, a model prediction basically anything R outputs. We can use that variable later simply by calling the name of the variable. 

To declare a variable, we need to asssign a variable name. The name should not have space. We can use `_ to connect to words. 

To add a value to the variable,  use `<-` or `=. 
Here is the syntax:

```
# First way to declare a variable:  use the <-
name_of_variable <- value
# Second way to declare a variable:  use the =
name_of_variable = value
```

In the command line, we can write the following codes to see what happens:


```r
# Print variable x
x <- 42
x
```

```
## [1] 42
```

```r
y  <- 10
y
```

```
## [1] 10
```

```r
# We call x and y and apply a subtraction
x-y
```

```
## [1] 32
```

### Basic data types

R works with numerous data types, including

- Scalars
- Vectors (numerical, character, logical)
- Matrices 
- Dataframes
- Lists

For instance, we can enumerate different types of data as follow:

-	`4.5` is a decimal value called **numerics**.
		`4` is a natural value called **integers**. Integers are also numerics.
		`TRUE` or `FALSE` are **Boolean** value called **logical**.
		Value inside `" "`  or `' ' are text (string). They are called **characters**.

We can check the type of a variable with the `class` function


```r
# Declare variables of different types
# Numeric
numeric <- 28
class(numeric)
```

```
## [1] "numeric"
```

```r
# String
character <- "R is Fantastic"
class(character)
```

```
## [1] "character"
```

```r
# Boolean
logical <- TRUE 
class(logical)
```

```
## [1] "logical"
```

**Vectors**

A vector is a one-dimeniosnal array. We can create a vector with all the basic data type we learnt before. 

The simplest way to build a vector in R, is to use the `c` command.


```r
# Numerical store
numeric_vector <- c(1, 10, 49)
numeric_vector
```

```
## [1]  1 10 49
```

```r
# Character store
character_vector <- c("a", "b", "c")
character_vector
```

```
## [1] "a" "b" "c"
```

```r
# Boolean store
boolean_vector <-  c(TRUE, FALSE, TRUE)
boolean_vector
```

```
## [1]  TRUE FALSE  TRUE
```

We can do arithmetic calculations on vectors.


```r
# Create the vectors
vect_1 <- c(1, 3, 5)
vect_2 <- c(2, 4, 6)

# Take the sum of A_vector and B_vector
sum_vect <- vect_1 + vect_2
  
# Print out total_vector
sum_vect
```

```
## [1]  3  7 11
```

In R, it is possible to slice a vector. In some occasion, we are interested in only the first five rows of a vector. We can use the `[1:5]` command to extract the value 1 to 5. 


```r
# Slice the first five rows of the vector
slice_vector <- c(1,2,3,4,5,6,7,8,9,10)
slice_vector[1:5]
```

```
## [1] 1 2 3 4 5
```

A shortest way to create a range of value is to use the `:` between two numbers. For instance, from the above example, we can write `c(1:10)` to create a vector of value from one to ten.


```r
# Faster way to create adjacent values
c(1:10)
```

```
##  [1]  1  2  3  4  5  6  7  8  9 10
```

### Logical Operators

With logical operators, we want to return values inside the vector based on logical conditions. Following is a detailed list of logical operators available in R

| Operator  | Description              |
|-----------|--------------------------|
| <         | Less than                |
| <=        | Less than or equal to    |
| >         | Greater than             |
| >=        | Greater than or equal to |
| ==        | Exactly equal to         |
| !=        | Not equal to             |
| !x        | Not x                    |
| x | y     | x OR y                   |
| x & y     | x AND y                  |
| isTRUE(x) | Test if X is TRUE        |

Logical statement in R are wrapped inside the `[]`. We can add a many conditional statement as we like but we need to include them inside parenthesis. We can follow this structure to create a conditional statement:

```
variable_name[(conditional_statement)]
```

with `variable_name` refering to the variable we want to use for the statement. We create the logical statement i.e. `variable_name > 0`. Finally, we use the square bracket to finalize the logical statement. Below, an example of logical statement.


```r
# Create a vector from 1 to 10
logical_vector <- c(1:10)
```

In the output above, R reads each value and compares it to the statement `logical_vector>5`. If the value is strickly superior to five, then the condition is `TRUE`, otherwise `FALSE`. R returns a vector of `TRUE` and `FALSE`. 

In the example below, we want to extract the values that only meet the condition *is strictly superior to five*. For that, we can wrap the condition inside a square bracket precede by the vector containing the values.


```r
# Print value stricly above 5
logical_vector[(logical_vector>5)]
```

```
## [1]  6  7  8  9 10
```

```r
# Print 5 and 6
logical_vector[(logical_vector>4) & (logical_vector<7)]
```

```
## [1] 5 6
```

### Matrix

A matrix is a 2 dimensional array that has m number of rows and n number of columns. In other words, matrix is a combination of two or more vectors with the same data type.  

*Note*: It is possible to create more than two dimensions arrays with R.

**Example of different matrix dimension**

- $2*2$ matrix

| 2x2 matrix | column 1 | column 2 |
|------------|----------|----------|
| row 1      | 1        | 2        |
| row 2      | 3        | 4        |

- $3*3$ matrix

| 3x3 matrix | column 1 | column 2 | Column 3 |
|------------|----------|----------|----------|
| row 1      | 1        | 2        | 3        |
| row 2      | 4        | 5        | 6        |
| row 3      | 7        | 8        | 9        |

- $5*2$ matrix

| 5x2 matrix | column 1 | column 2 |
|------------|----------|----------|
| row 1      | 1        | 2        |
| row 2      | 3        | 4        |
| row 3      | 5        | 6        |
| row 4      | 7        | 8        |
| row 5      | 9        | 10       |

We can create a matrix with the function `matrix()`. This function takes three arguments: 

```
matrix(data, nrow, ncol, byrow = FALSE)
Arguments:

- data: The collection of elements that R will arrange into the rows and columns of the matrix
- nrow: Number of rows
- ncol: Number of columns
- byrow: The rows are filled from the left to the right. We use `byrow = FALSE` (default values), if we want the matrix to be filled by the columns i.e. the values are filled top to bottom.
```

Let's construct a 5x2 matrix with a sequence of number from 1 to 10. We will create two separate matrix, one with `byrow =TRUE` and one with `byrow =  FALSE to see the difference


```r
# Construct a matrix with 5 rows that contain the numbers 1 up to 10 and byrow =  TRUE
matrix_a <-matrix(1:10, byrow = TRUE, nrow = 5)
matrix_a
```

```
##      [,1] [,2]
## [1,]    1    2
## [2,]    3    4
## [3,]    5    6
## [4,]    7    8
## [5,]    9   10
```

```r
# Print dimension of the matrix with dim()
dim(matrix_a)
```

```
## [1] 5 2
```

```r
# Construct a matrix with 5 rows that contain the numbers 1 up to 10 and byrow =  FALSE
matrix_a_1 <-matrix(1:10, byrow = FALSE, nrow = 5)
matrix_a_1
```

```
##      [,1] [,2]
## [1,]    1    6
## [2,]    2    7
## [3,]    3    8
## [4,]    4    9
## [5,]    5   10
```

```r
# Print dimension of the matrix with dim()
dim(matrix_a_1)
```

```
## [1] 5 2
```

You can also create a 4x3 matrix using `ncol`.R will create 3 columns and fill the row from top to bottom. Check an example


```r
matrix_a_2 <-matrix(1:12, byrow = FALSE, ncol = 3)
matrix_a_2
```

```
##      [,1] [,2] [,3]
## [1,]    1    5    9
## [2,]    2    6   10
## [3,]    3    7   11
## [4,]    4    8   12
```

```r
dim(matrix_a_2)
```

```
## [1] 4 3
```

You can add a column to a matrix with the `cbind()` command. `cbind()` means column binding. `cbind()` concanates as many matrix or columns as specified. For example, our previous example created a 5x2 matrix. We concanate a third column and verify the dimension is 5x3.


```r
# Concanate c(1:5) to the matrix_a
matrix_b <- cbind(matrix_a, c(1:5))

# Check the dimension
dim(matrix_b)
```

```
## [1] 5 3
```

We can also add more than one column. Let's see the next sequence of number to the `matrix_a_2` matrix. The dimension of the number matrix will be 4x6 with number from 1 to 24.


```r
matrix_a_3 <-matrix(13:24, byrow = FALSE, ncol = 3)
matrix_a_4 <- cbind(matrix_a_2, matrix_a_3)
dim(matrix_a_4)
```

```
## [1] 4 6
```

*Note*: The number of rows of matrices should be equal for cbind work

`cbind()` concanates columns, `rbind()` appends rows. Let's add one row to our `matrix_b` matrix and verify the dimension is 6x3


```r
# Create a vector of 3 columns
add_row <- c(1:3)

# Append to the matrix
matrix_c <- rbind(matrix_b, add_row)

# Check the dimension
dim(matrix_c)
```

```
## [1] 6 3
```

We can select elements one or many elements from a matrix by using the square brackets `[ ]`. This is where slicing comes into picture.

For example:

-	`matrix_c[1,2]` selects the element at the first row and second column.
- `matrix_c[1:3,2:3]` results in a matrix with the data on the rows 1, 2, 3 and columns 2, 3, 
- `matrix_c[,1]` selects all elements of the first column.
- `matrix_c[1,]` selects all elements of the first row.

Here is the output you get for the above codes


```r
library(dplyr)
data_frame <- tibble(
  c1 = rnorm(50, 5, 1.5), 
  c2 = rnorm(50, 5, 1.5),
  c3 = rnorm(50, 5, 1.5),
  c4 = rnorm(50, 5, 1.5),
  c5 = rnorm(50, 5, 1.5)
)

# return the first value of the first column
data_frame[1:1]
```

```
## # A tibble: 50 x 1
##       c1
##    <dbl>
##  1  5.03
##  2  4.76
##  3  3.05
##  4  5.08
##  5  4.72
##  6  5.10
##  7  4.44
##  8  5.52
##  9  4.43
## 10  4.97
## # ... with 40 more rows
```

```r
# return a matrix with the values on the rows  and 2 and columns 1 and 2.
data_frame[1:2,1:2] 
```

```
## # A tibble: 2 x 2
##      c1    c2
##   <dbl> <dbl>
## 1  5.03  5.05
## 2  4.76  6.65
```

```r
# return the first column with all values.
data_frame[,2] 
```

```
## # A tibble: 50 x 1
##       c2
##    <dbl>
##  1  5.05
##  2  6.65
##  3  7.88
##  4  4.83
##  5  3.24
##  6  6.54
##  7  3.47
##  8  3.84
##  9  5.65
## 10  7.68
## # ... with 40 more rows
```

```r
# return the first row.
data_frame[1,] 
```

```
## # A tibble: 1 x 5
##      c1    c2    c3    c4    c5
##   <dbl> <dbl> <dbl> <dbl> <dbl>
## 1  5.03  5.05  5.23  4.33  4.17
```

### Categorical and Continuous types of variable

In a dataset, we can distinguish two types of variables: **categorical** and **continuous**.

-	In a categorical variable, the value is limited, and usually based on a particular finite group. For example, a categorical variable can be countries, year, gender, occupation. 
		A continuous variable, however, can take any values, from integer to decimal. For example, we can have the revenue, price of a share, etc.. 

**Categorical variables**

R stores categorical variables into a factor. Let's check the code below to convert a character variable into a factor variable. **Characters** are not supported in machine learning algorithm, and the only way is to convert a string to an integer.

```
factor(x = character(), levels, labels = levels, ordered = is.ordered(x))
Arguments:

- x: A vector of data. Need to be a string or integer, not decimal.
- levels: A vector of possible values taken by x. This argument is optional, and default value is the unique list of items of the vector x.
- labels: Add a label to the x data. For example, 1 can take the label `male` while 0, the label `female`. 
ordered: Determine if the levels should be ordered.
```

Let's create a factor dataframe.


```r
# Create gender vector
gender_vector <- c("Male", "Female", "Female", "Male", "Male")
class(gender_vector)
```

```
## [1] "character"
```

```r
# Convert gender_vector to a factor
factor_gender_vector <-factor(gender_vector)
class(factor_gender_vector)
```

```
## [1] "factor"
```

It is important to transform a **string** into factor when we perform Machine Learning task. A factor column allows R to transform the matrix into a **one hot encoder**. For instance, in a case of a two dimensions factor, R will create behind the hood 2 columns (i.e. if `male` then the matrix is `[1,0,0,1,1]`, if `female` then `[0,1,1,0,0]`)

There are two types of categorical variables: a **nominal categorical variable** and an **ordinal categorical variable**.

**Nominal categorical variable**

A categorical variable has several values but the order does not matter. For instance, male or female categorical variable do not have ordering.


```r
# Create a color vector
color_vector <- c('blue', 'red', 'green', 'white', 'black', 'yellow')

# Convert the vector to factor
factor_color <- factor(color_vector)
factor_color
```

```
## [1] blue   red    green  white  black  yellow
## Levels: black blue green red white yellow
```

From the `factor_color`, we can't tell any order.

**Ordinal categorical variable**

Ordinal categorical variables do have a natural ordering. We can specify the order, from the lowest to the highest with `order = TRUE` and highest to lowest with `order = FALSE`. 

We can use `summary` to count the values for each factor.


```r
# Create Ordinal categorical vector 
day_vector <- c('evening', 'morning', 'afternoon', 'midday', 'midnight', 'evening')

# Convert `day_vector` to a factor with ordered level
factor_day <- factor(day_vector, 
                     order = TRUE,
                     levels =c('morning', 'midday', 'afternoon', 'evening', 'midnight'))

# Print the new variable
factor_day
```

```
## [1] evening   morning   afternoon midday    midnight  evening  
## Levels: morning < midday < afternoon < evening < midnight
```

```r
# Count the number of occurence of each level
summary(factor_day)
```

```
##   morning    midday afternoon   evening  midnight 
##         1         1         1         2         1
```

R ordered the level from 'morning' to 'midnight' as specified in the `levels` parenthesis.

We can create an ordering level with label for the column male and female, with 1 refers to `female` and 0 for `male`.


```r
set.seed(1)
# Create a random vector of 0 and 1
gender <- sample(0:1, 100, replace=T)

# Create a factor variable with the label option
factor_gender <- factor(gender, 
                     ordered = TRUE,
                     labels =c('Male', 'Female'))
head(factor_gender)                    
```

```
## [1] Male   Male   Female Female Male   Female
## Levels: Male < Female
```

**Continuous variables**

Continuous class variables are the default value in R. They are stored as numeric or integer. We can see it from the dataset below. `mtcars` is a built in dataset. It gathers information on different types of car. We can import it by using `mtcars` and check the class of the variable `mpg`, mile per gallon. It returns a numeric values, indicating a continuous variable.


```r
dataset <- mtcars
class(dataset$mpg)
```

```
## [1] "numeric"
```

