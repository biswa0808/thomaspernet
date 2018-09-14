---
title: Data Frame
author: Thomas
date: '2018-08-11'
slug: data-frame
categories: []
tags:
  - intro
header:
  caption: ''
  image: ''
---

### Object in R

### Data frame

A **data frame** is a list of vectors which are of equal length. A matrix contains only one type of data, while a data frame accepts different data types (numeric, character, factor, etc.).

**Create a data frame**

You can create a data frame by passing the variable `a,b,c,d` into the `data.frame()` function. you can name the columns with `name()` and simply specify the name of the variables. 

```
data.frame(df), stringsAsFactors = TRUE)
arguments:

- df: It can be a matrix to convert as a data frame or a collection of variables to join
- stringsAsFactors: Convert string to factor by default
```

you can create our first data set by combining four variables of same length.


```r
# Create a, b, c, d variables
a <- c(10,20,30,40)
b <- c('book', 'pen', 'textbook', 'pencil_case')
c <- c(TRUE,FALSE,TRUE,FALSE)
d <- c(2.5, 8, 10, 7)

# Join the variables to create a data frame
df <- data.frame(a,b,c,d)
df
```

```
##    a           b     c    d
## 1 10        book  TRUE  2.5
## 2 20         pen FALSE  8.0
## 3 30    textbook  TRUE 10.0
## 4 40 pencil_case FALSE  7.0
```

you can see the column headers have the same name as the variables. you can change the column name with the function `names()`. Check the example below:


```r
# Name the data frame
names(df) <- c('ID', 'items', 'store', 'price')
df
```

```
##   ID       items store price
## 1 10        book  TRUE   2.5
## 2 20         pen FALSE   8.0
## 3 30    textbook  TRUE  10.0
## 4 40 pencil_case FALSE   7.0
```

```r
# Print the structure
str(df)
```

```
## 'data.frame':	4 obs. of  4 variables:
##  $ ID   : num  10 20 30 40
##  $ items: Factor w/ 4 levels "book","pen","pencil_case",..: 1 2 4 3
##  $ store: logi  TRUE FALSE TRUE FALSE
##  $ price: num  2.5 8 10 7
```

*Note*, by default, data frame return string variables as factor.

**Slice a factor**

It is possible to **slice** values of a data frame. you select the rows and columns to return into bracket precede by the name of the data frame.

A data frame is composed by rows and columns, `df[A, B]`. `A` represents the rows and `B` the columns. you can slice either by specifying which rows or columns. 

From picture 1, the left part represents the **rows** and the right part is the **columns**. Note that the symbol `:` means **to**. For instance, `1:3` intends to select values from 1 **to** 3.

<img src="/project/data-types_files/1.png" width="55%" style="display: block; margin: auto;" />

In below diagram you display how to access different selection of the data frame:

- The yellow arrow selects the **row** 1 in **column** 2
- The green arrow selects the **rows** 1 to 2 
- The red arrow selects the **column** 1
- The blue arrow selects the **rows** 1 to 3 and **columns** 3 to 4

Note that, if you let the left part blank, R will select **all the rows**. By analogy, if you let the right part blank, R will select **all the columns**.

<img src="/project/data-types_files/2.png" width="55%" style="display: block; margin: auto;" />

you can run the code in the console: 


```r
## Select row 1 in column 2 

df[1,2]
```

```
## [1] book
## Levels: book pen pencil_case textbook
```

```r
## Select Rows 1 to 2

df[1:2,]
```

```
##   ID items store price
## 1 10  book  TRUE   2.5
## 2 20   pen FALSE   8.0
```

```r
## Select Columns 1

df[,1]
```

```
## [1] 10 20 30 40
```

```r
## Select Rows 1 to 3 and columns 3 to 4

df[1:3, 3:4]
```

```
##   store price
## 1  TRUE   2.5
## 2 FALSE   8.0
## 3  TRUE  10.0
```

It is also possible to select the columns with their names. For instance, the code below extracts two columns:  `ID` and `store`.


```r
# Slice with columns name
df[, c('ID', 'store')]
```

```
##   ID store
## 1 10  TRUE
## 2 20 FALSE
## 3 30  TRUE
## 4 40 FALSE
```

**Append a Column to Data Frame**

You can also append a column to a Data Frame. You need to use symbol `$ to append a new variable.


```r
# Create a new vector
quantity <- c(10, 35, 40, 5)

# Add `quantity` to the `df` data frame
df$quantity <- quantity

df
```

```
##   ID       items store price quantity
## 1 10        book  TRUE   2.5       10
## 2 20         pen FALSE   8.0       35
## 3 30    textbook  TRUE  10.0       40
## 4 40 pencil_case FALSE   7.0        5
```

*Note*: The number of elements in the vector has to be equal to the no of elements in data frame.  Executing the following statement 


```r
quantity <- c(10, 35, 40)
# Add `quantity` to the `df` data frame
df$quantity <- quantity
```

**Select a column of a data frame**

Sometimes, we need to store a column of a data frame for future use or perform operation on a column. We can use the $ sign to select the column from a data frame.


```r
# Select the column ID 
select_id <- df$ID
```

**Subset a data frame**

In the previous section, we selected an entire column without condition. It is possible to **subset** based on whether or not a certain condition was true. 

You use the `subset()` function. 

```
subset(x, condition)
arguments:

- x: data frame used to perform the subset
- condition: define the conditional statement
```

You want to return only the items with `quantity` above 10, you can do : 


```r
# Select quantity above 10
subset(df, subset = quantity > 10)
```

```
##   ID    items store price quantity
## 2 20      pen FALSE     8       35
## 3 30 textbook  TRUE    10       40
```

### List

A **list** is a great tool to store many kinds of object in the order expected. We can include matrices, vectors data frames or lists. We can imagine a list as a bag in which we want to put many different items. When we need to use an item, we open the bag and use it. A list is similar, we can store a collection of objects and use them when we need them. 

You can use `list() function to create a list.

```
list(element_1, ...)
arguments:

- element_1: store any type of R object
- ...: pass as many object as specifying. each object needs to be separated by a comma
```

In the example below, you create three different objects, a vector, a matrix and a data frame. 


```r
# Vector with numerics from 1 up to 5
vect  <- 1:5 

# A 2x 5 matrix
mat  <- matrix(1:9, ncol = 5)
dim(mat)
```

```
## [1] 2 5
```

```r
# select the 10th row of the built-in R data set EuStockMarkets
df <- EuStockMarkets[1:10,]
```

Now, you can put the three objects into a list.


```r
# Construct list with these vec, mat and df:
my_list <- list(vect, mat, df)
my_list
```

```
## [[1]]
## [1] 1 2 3 4 5
## 
## [[2]]
##      [,1] [,2] [,3] [,4] [,5]
## [1,]    1    3    5    7    9
## [2,]    2    4    6    8    1
## 
## [[3]]
##           DAX    SMI    CAC   FTSE
##  [1,] 1628.75 1678.1 1772.8 2443.6
##  [2,] 1613.63 1688.5 1750.5 2460.2
##  [3,] 1606.51 1678.6 1718.0 2448.2
##  [4,] 1621.04 1684.1 1708.1 2470.4
##  [5,] 1618.16 1686.6 1723.1 2484.7
##  [6,] 1610.61 1671.6 1714.3 2466.8
##  [7,] 1630.75 1682.9 1734.5 2487.9
##  [8,] 1640.17 1703.6 1757.4 2508.4
##  [9,] 1635.47 1697.5 1754.0 2510.5
## [10,] 1645.89 1716.3 1754.3 2497.4
```

**Select elements from list**

After we built our list, we can access it quite easily. We need to use the `[[index]] to select an element in a list. The value inside the double square bracket represents the position of the item in a list we want to extract. For instance, we pass 2 inside the parenthesis, R returns the second element listed.


Let's try to select the second items of the list named `my_list`, we use `my_list[[2]]`.


```r
# Print second element of the list
my_list[[2]]
```

```
##      [,1] [,2] [,3] [,4] [,5]
## [1,]    1    3    5    7    9
## [2,]    2    4    6    8    1
```

**Build-in data frame**

Before to create our own data frame, we can have a look at the R data set available online. The prison dataset is a 714x5 dimension. We can get a quick look at the top of the data frame with `head()` function. By analogy, `tail()` displays the bottom of the data frame. You can specify the number of rows shown with `head (df, 5)`. We will learn more about the function `read.csv() in future tutorial.

 

```r
# Print the head of the data
library(dplyr)
PATH <- 'https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/prison.csv'
df <- read.csv(PATH)[1:5]
head(df, 5)
```

```
##   X state year govelec  black
## 1 1     1   80       0 0.2560
## 2 2     1   81       0 0.2557
## 3 3     1   82       1 0.2554
## 4 4     1   83       0 0.2551
## 5 5     1   84       0 0.2548
```

you can check the structure of the data frame with `str`:


```r
# Structure of the data
str(df)
```

```
## 'data.frame':	714 obs. of  5 variables:
##  $ X      : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ state  : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ year   : int  80 81 82 83 84 85 86 87 88 89 ...
##  $ govelec: int  0 0 1 0 0 0 1 0 0 0 ...
##  $ black  : num  0.256 0.256 0.255 0.255 0.255 ...
```

All variables are stored in the *numerical* format.


