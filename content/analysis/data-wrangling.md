---
title: Data Wrangling
author: Thomas
date: []
slug: data-wrangling
categories: []
tags:
  - preparation
header:
  caption: ''
  image: ''
---

# Introduction to Data Preparation

Data preparation can be divided into three parts

-	**Extraction**: First, you need to collect the data from many sources and combine them.
-	**Transform**: This step involves the data manipulation. Once you have consolidated all the sources of data, you can begin to clean the data. 
-	**Visualize**: The last move is to visualize your data to check irregularity.

One of the most significant challenges faced by data scientist is the data manipulation. Data is never available in the desired format. The data scientist needs to spend at least half of his time, cleaning and manipulating the data. That is one of the most critical assignments in the job. If the data manipulation process is not complete, precise and rigorous, the model will not perform correctly.

R has a library called `dplyr` to help in data transformation. The `dplyr` library is fundamentally created around four functions to manipulate the data and five verbs to clean the data. After that, you can use the ggplot library to analyze and visualize the data.

In this chapter, you will learn how to use the `dplyr` library to manipulate a data frame. 

![](/project/data-wrangling_files/16.png)

You need to install `dplyr` library:

- `caret`: Data manipulation library. If you have install R with `r-essential`. It is already in the library
    - [Anaconda](https://anaconda.org/r/r-dplyr): `conda install -c r r-dplyr`
    
## Data Wrangling

### Merge with `dplyr()`

`dplyr` provides a nice and convenient way to combine datasets. You may have many sources of input data, and at some point, you need to combine them. A join with `dplyr` adds variables to the right of the original dataset. The beauty is `dplyr` is that it handles four types of joins similar to SQL

- `Left_join()`
- `right_join()`
- `inner_join()`
- `full_join()`

You will study all the joins types via an easy example. 

First of all, you build two datasets. Table 1 contains two variables, `ID` and `y`, whereas table 2 gathers `ID` and `z`. In each situation, you need to have a **key-pair** variables. In your case, `ID` is your **key** variable. The function will look for identical values in both tables and binds the returning values to the right of table 1. 

![](/project/data-wrangling_files/9.png)


```r
library(dplyr)
df_primary <- tribble(
  ~ID, ~y, 
  "A", 5,
  "B", 5, 
  "C", 8, 
  "D", 0,
  "F", 9
)

df_secondary <- tribble(
  ~ID, ~y, 
  "A", 30,
  "B", 21, 
  "C", 22,  
  "D", 25,
  "E", 29
)
```

**Function `left_join()`**

The most common way to merge two datasets is to use the `left_join()` function. You can see from the picture below that the key-pair matches perfectly the rows `A`, `B`, `C` and `D` from both datasets. However, `E` and `F` are left over. How do you treat these two observations?. With the `left_join()`, you will keep all the variables in the original table and don't consider the variables that do not have a key-paired in the destination table. In your example, the variable `E` does not exist in table 1. Therefore, the row will be dropped. The variable `F` comes from the origin table; it will be kept after the `left_join()` and return `NA` in the column `z`. The figure below reproduces what will happen with a `left_join()`.

![](/project/data-wrangling_files/10.png)


```r
left_join(df_primary, df_secondary, by = 'ID')
```

```
## # A tibble: 5 x 3
##   ID      y.x   y.y
##   <chr> <dbl> <dbl>
## 1 A        5.   30.
## 2 B        5.   21.
## 3 C        8.   22.
## 4 D        0.   25.
## 5 F        9.   NA
```

**Function `right_join()`**

The `right_join()` function works exactly like `left_join()`. The only difference is the row dropped. The value `E`, available in the destination, exists in the new table and takes the value `NA` for the column `y`.

![](/project/data-wrangling_files/11.png)


```r
right_join(df_primary, df_secondary, by = 'ID')
```

```
## # A tibble: 5 x 3
##   ID      y.x   y.y
##   <chr> <dbl> <dbl>
## 1 A        5.   30.
## 2 B        5.   21.
## 3 C        8.   22.
## 4 D        0.   25.
## 5 E       NA    29.
```

**Function `inner_join()`**

When you are 100% sure that the two datasets won't match, you can consider to return **only** rows existing in **both** dataset. This is possible when you need a clean dataset or when you don't want to impute missing values with the mean or median. 

The `inner_join()`comes to help. This function excludes the unmatched rows. 

![](/project/data-wrangling_files/12.png)


```r
inner_join(df_primary, df_secondary, by = 'ID')
```

```
## # A tibble: 4 x 3
##   ID      y.x   y.y
##   <chr> <dbl> <dbl>
## 1 A        5.   30.
## 2 B        5.   21.
## 3 C        8.   22.
## 4 D        0.   25.
```

**Function `full_join()`**

Finally, the `full_join()` function keeps all observations and replace missing values with `NA`.

![](/project/data-wrangling_files/13.png)


```r
full_join(df_primary, df_secondary, by = 'ID')
```

```
## # A tibble: 6 x 3
##   ID      y.x   y.y
##   <chr> <dbl> <dbl>
## 1 A        5.   30.
## 2 B        5.   21.
## 3 C        8.   22.
## 4 D        0.   25.
## 5 F        9.   NA 
## 6 E       NA    29.
```

**Multiple keys pairs**

Last but not least, you can have multiple keys in your dataset. Consider the following dataset where you have years or a list of products bought by the customer. 

![](/project/data-wrangling_files/14.png)

 If you try to merge both table, R throws an error. To remedy the situation, you can pass two key-pairs variables. That is, `ID` and `year`appear in both datasets. You can use the following code to merge table1 and table 2.


```r
df_primary <- tribble(
  ~ID, ~year, ~items,
  "A", 2015,3,
  "A", 2016,7 ,
  "A", 2017, 6,
  "B", 2015,4,
  "B", 2016, 8,
  "B", 2017,7,
  "C", 2015, 4,
  "C", 2016, 6,
  "C", 2017, 6
)

df_secondary <- tribble(
  ~ID, ~year, ~prices,
  "A", 2015,9,
  "A", 2016,8 ,
  "A", 2017,12,
  "B", 2015,13,
  "B", 2016, 14,
  "B", 2017,6,
  "C", 2015, 15,
  "C", 2016, 15,
  "C", 2017, 13
)

left_join(df_primary, df_secondary, by = c('ID', 'year'))
```

```
## # A tibble: 9 x 4
##   ID     year items prices
##   <chr> <dbl> <dbl>  <dbl>
## 1 A     2015.    3.     9.
## 2 A     2016.    7.     8.
## 3 A     2017.    6.    12.
## 4 B     2015.    4.    13.
## 5 B     2016.    8.    14.
## 6 B     2017.    7.     6.
## 7 C     2015.    4.    15.
## 8 C     2016.    6.    15.
## 9 C     2017.    6.    13.
```

### Data transformation

Following are four important functions to tidy the data:

- `gather()`: Transform the data from wide to long
- `spread()`: Transform the data from long to wide
- `separate()`: Split one variables into two
- `unit()`: Unit two variables into one

You use the `tidyr` library. This library belongs to the collection of library to manipulate, clean and visualize the data. 

- `tidyr `: Manipulate data frame. If you have install R with `r-essential`. It is already in the library
    - [Anaconda](https://anaconda.org/r/r-tidyr): `conda install -c r r-tidyr`

Instead, you can use the console: 

`install.packages("tidyr")`

to install `tidyr`

**Function `gather()`**

The objectives of the `gather()` function is to transform the data from wide to long.

```
gather(data, key, value, na.rm = FALSE)
arguments:
  
- data: The data frame used to reshape the dataset 
- key: Name of the new column created
- value: Select the columns used to fill the key column
- na.rm: Remove missing values. FALSE by default
```

Below, you can visualize the concept of reshaping wide to long. You want to create a single column names `growth`, filled by the rows of the `quarter` variables.

![](/project/data-wrangling_files/15.png)


```r
library(tidyr)
# Create messy dataset
messy <- data.frame(
  country = c("A", "B", "C"),
  q1_2017 = c(0.03, 0.05, 0.01),
  q2_2017 = c(0.05, 0.07, 0.02), 
  q3_2017 = c(0.04, 0.05, 0.01), 
  q4_2017 = c(0.03, 0.02, 0.04)
)
messy
```

```
##   country q1_2017 q2_2017 q3_2017 q4_2017
## 1       A    0.03    0.05    0.04    0.03
## 2       B    0.05    0.07    0.05    0.02
## 3       C    0.01    0.02    0.01    0.04
```

```r
# Reshape the data
tidier <- messy %>%
          gather(quarter, growth, q1_2017:q4_2017)
tidier
```

```
##    country quarter growth
## 1        A q1_2017   0.03
## 2        B q1_2017   0.05
## 3        C q1_2017   0.01
## 4        A q2_2017   0.05
## 5        B q2_2017   0.07
## 6        C q2_2017   0.02
## 7        A q3_2017   0.04
## 8        B q3_2017   0.05
## 9        C q3_2017   0.01
## 10       A q4_2017   0.03
## 11       B q4_2017   0.02
## 12       C q4_2017   0.04
```

In the `gather()` function, you create two new variable `quarter` and `growth` because your original dataset has one group variable: i.e. `country` and the key-value pairs. 

**Function `spread()`**

the `spread()` function does the opposite of gather. 

```
spread(data, key, value)
arguments:
  
- data: The data frame used to reshape the dataset 
- key: Column to reshape long to wide
- value: Rows used to fill the new column
```

You can reshape the `tidier` dataset back to `messy` with `spread()


```r
# Reshape the data
messy_1 <- tidier %>%
          spread(quarter, growth)
messy_1
```

```
##   country q1_2017 q2_2017 q3_2017 q4_2017
## 1       A    0.03    0.05    0.04    0.03
## 2       B    0.05    0.07    0.05    0.02
## 3       C    0.01    0.02    0.01    0.04
```

**Function `separate()`**

The `separate()` function splits a column into two according to a separator. This function is helpful in some situation where the variable is a date. Our analysis can require focusing on month and year and you want to separate the column into two new variables. 

```
separate(data, col, into, sep= "", remove = TRUE)
arguments:

- data: The data frame used to reshape the dataset 
- col: The column to split
- into: The name of the new variables
- sep: Indicates the symbol used that unit the variable name, i.e:  "-", "_", "&"
remove: Remove the old column. By default sets to TRUE.
```

You can split the quarter from the year in the `tidier` dataset by applying the `separate()` function. 


```r
separate_tidier <- tidier %>%
                   separate(quarter, c("Qrt", "year"), sep ="_")
head(separate_tidier)
```

```
##   country Qrt year growth
## 1       A  q1 2017   0.03
## 2       B  q1 2017   0.05
## 3       C  q1 2017   0.01
## 4       A  q2 2017   0.05
## 5       B  q2 2017   0.07
## 6       C  q2 2017   0.02
```

**Function `unite()`**

The `unite()` function concanates two columns into one. 

```
unit(data, col, conc ,sep= "", remove = TRUE)
arguments:

- data: The data frame used to reshape the dataset 
- col: Name of the new colunm
- conc: Name of the columns to concanate
- sep: Indicates the symbol used that unit the variable name, i.e: "-", "_", "&"
- remove: Remove the old columns. By default sets to TRUE
```

In the above example, you separated quarter from year. What if you want to merge them. You use the following code:


```r
unit_tidier <- separate_tidier %>%
                   unite(Quarter, Qrt, year, sep ="_")
head(unit_tidier)
```

```
##   country Quarter growth
## 1       A q1_2017   0.03
## 2       B q1_2017   0.05
## 3       C q1_2017   0.01
## 4       A q2_2017   0.05
## 5       B q2_2017   0.07
## 6       C q2_2017   0.02
```

### Summary 

The table below summarises the four functions used in `dplyr` to merge two datasets. 

| Library | Objectives                                                           | Function     | Arguments                            | Multiple keys                            |
|---------|----------------------------------------------------------------------|--------------|--------------------------------------|------------------------------------------|
| Dplyr   | Merge two datasets. Keep all observations from the origin table      | left_join()  | data, origin, destination, by = "ID" | origin, destination, by = c("ID", "ID2") |
| Dplyr   | Merge two datasets. Keep all observations from the destination table | right_join() | data, origin, destination, by = "ID" | origin, destination, by = c("ID", "ID2") |
| Dplyr   | Merge two datasets. Excludes all unmatched rows                      | inner_join() | data, origin, destination, by = "ID" | origin, destination, by = c("ID", "ID2") |
| Dplyr   | Merge two datasets. Keeps all observations                           | full_join()  | data, origin, destination, by = "ID" | origin, destination, by = c("ID", "ID2") |

The last table summarises how to transform a dataset with the `gather()`, `spread()`, `separate()` and `unit()` functions.

| Library | Objectives                           | Function   | Arguments                                 |
|---------|--------------------------------------|------------|-------------------------------------------|
| tidyr   | Transform the data from wide to long | gather()   | (data, key, value, na.rm = FALSE)         |
| tidyr   | Transform the data from long to wide | spread()   | (data, key, value)                        |
| tidyr   | Split one variables into two         | separate() | (data, col, into, sep= "", remove = TRUE) |
| tidyr   | Unit two variables into one          | unit()     | (data, col, conc ,sep= "", remove = TRUE) |

