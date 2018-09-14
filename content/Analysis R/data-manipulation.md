---
title: Data Manipulation
author: Thomas
date: []
slug: data-manipulation
categories: []
tags:
  - intro
header:
  caption: ''
  image: ''
---

## Data Manipulation

### Merge data

Very often, you have data from multiple sources. To perform an analysis, you need to **merge** two data frames together with one or more **common key variables**. 

**Full match**

A full match returns values that have a counterpart in the destination table. The values that are not match won't be return in the new data frame. The partial match,  however, return the missing values as  `NA`.

You will see a simple **inner join**. The inner join keyword selects records that have matching values in both tables. To join two datasets, you can use `merge()` function. You will use three arguments :

```
merge(x, y, by.x = bx, by.y = y)
Arguments:

- x: The origin data frame
- y: The data frame to merge
- by.x: The column used for merging in x data frame. Column x to merge on
- by.y: The column used for merging in y data frame. Column y to merge on
```

Example: 
Create First Dataset with variables

-	surname
-	nationality

Create Second Dataset with variables

-	surname
-	movies

The common key variable is surname. You can merge both data and check if the dimensionality is 7x3.

You add `stringsAsFactors=FALSE` in the data frame because you don't want R to convert string as factor, you want the variable to be treated as character.


```r
# Create origin dataframe
producers <- data.frame(
    surname = c("Spielberg", "Scorsese", "Hitchcock", "Tarantino", "Polanski"),
    nationality = c("US", "US", "UK", "US", "Poland"),    stringsAsFactors=FALSE)

# Create destination dataframe
movies <- data.frame(
    surname = c("Spielberg", "Scorsese", "Hitchcock",
             "Hitchcock", "Spielberg", "Tarantino", "Polanski"),
    title = c("Super 8",
              "Taxi Driver",
              "Psycho",
              "North by Northwest", 
              "Catch Me If You Can",
              "Reservoir Dogs",
              "Chinatown"), 
               stringsAsFactors=FALSE)

# Merge two datasets
m1 <- merge(producers, movies, by.x = "surname")
dim(m1)
```

```
## [1] 7 3
```

Let's merge data frames when the common key variables have different names. 

You change surname to name in the movies data frame. You use the function identical(x1, x2) to check if both dataframes are identical.



```r
# Change name of `books` dataframe
colnames(movies)[colnames(movies) == 'surname'] <- 'name'

# Merge with different key value
m2 <- merge(producers, movies, by.x = "surname", by.y = "name")

# Print head of the data
head(m2)
```

```
##     surname nationality               title
## 1 Hitchcock          UK              Psycho
## 2 Hitchcock          UK  North by Northwest
## 3  Polanski      Poland           Chinatown
## 4  Scorsese          US         Taxi Driver
## 5 Spielberg          US             Super 8
## 6 Spielberg          US Catch Me If You Can
```

```r
# Check if data are identical
identical(m1, m2)
```

```
## [1] TRUE
```

**Partial match**

It is not surprising that two data frames do not have the same common key variables. In the **full matching**, the data frame returns **only** rows found in both x and y data frame. With **partial merging**, it is possible to keep the rows with no matching rows in the other data frame. These rows will have `NA`s in those columns that are usually filled with values from y. You can do that by setting `all.x= TRUE`. 

For instance, you can add a new producer, Lucas, in the producer data frame without the movie references in movies data frame. If you set `all.x= FALSE`, R will join only the matching values in both data set. In your case, the producer Lucas will not be join to the merge because it is missing from one dataset. 
 
Let's see the dimension of each output when you specify `all.x= TRUE` and when you don't. 


```r
# Create a new producer
add_producer <-  c('Lucas', 'US')

#  Append it to the `producer` dataframe
producers <- rbind(producers, add_producer)

# Use a partial merge 
m3 <-merge(producers, movies, by.x = "surname", by.y = "name", all.x = TRUE)
m3
```

```
##     surname nationality               title
## 1 Hitchcock          UK              Psycho
## 2 Hitchcock          UK  North by Northwest
## 3     Lucas          US                <NA>
## 4  Polanski      Poland           Chinatown
## 5  Scorsese          US         Taxi Driver
## 6 Spielberg          US             Super 8
## 7 Spielberg          US Catch Me If You Can
## 8 Tarantino          US      Reservoir Dogs
```

```r
# Use a full merge
m4 <-merge(producers, movies, by.x = "surname", by.y = "name", all.x = FALSE)
m4
```

```
##     surname nationality               title
## 1 Hitchcock          UK              Psycho
## 2 Hitchcock          UK  North by Northwest
## 3  Polanski      Poland           Chinatown
## 4  Scorsese          US         Taxi Driver
## 5 Spielberg          US             Super 8
## 6 Spielberg          US Catch Me If You Can
## 7 Tarantino          US      Reservoir Dogs
```

```r
# Compare the dimension of each data frame
dim(m1)
```

```
## [1] 7 3
```

```r
dim(m2)
```

```
## [1] 7 3
```

```r
dim(m3)
```

```
## [1] 8 3
```

```r
dim(m4)
```

```
## [1] 7 3
```

As you can see, the dimension of the new data frame 8x3 compare with 7x3 for `m1` and `m2`. R includes `NA` for the missing producer in the `producer` data frame. 

### Sort a data frame

In data analysis you can **sort** your data according to a certain variable in the dataset. In R, we can use the help of the function `order()`. In R, we can easily sort a vector of continuous variable or factor variable. Arranging the data can be of **ascending** or **descending** order. The syntax is:

```
sort(x, decreasing = FALSE, na.last = TRUE):
Argument:
  
- x: A vector containing continuous or factor variable
- decreasing: Control for the order of the sort method. By default, decreasing is set to  `FALSE`.
- na.last: Indicates whether the `NA` 's value should be put last or not
```

For instance, we can create a **tibble** data frame and sort one or multiple variables. A tibble data frame is a new approach to data frame. It improves the syntax of data frame and avoid frustrating data type formatting, especially for character to factor. It is also a convenient way to create a data frame by hand, which is our purpose here. To learn more about tibble, please refer to the [vignette]( https://cran.r-project.org/web/packages/tibble/vignettes/tibble.html)


```r
library(dplyr)
data_frame <- tibble(
  c1 = rnorm(50, 5, 1.5), 
  c2 = rnorm(50, 5, 1.5),
  c3 = rnorm(50, 5, 1.5),
  c4 = rnorm(50, 5, 1.5),
  c5 = rnorm(50, 5, 1.5)
)

# Sort by c1
df <-data_frame[order(data_frame$c1),]

# Sort by c3 and c4
df <-data_frame[order(data_frame$c3, data_frame$c4),]
head(df)
```

```
## # A tibble: 6 x 5
##      c1    c2    c3    c4    c5
##   <dbl> <dbl> <dbl> <dbl> <dbl>
## 1  3.62  5.76  2.46  2.28  5.72
## 2  1.64  6.24  2.50  5.76  4.52
## 3  5.58  4.93  2.62  5.36  6.01
## 4  3.87  7.34  2.69  3.44  7.06
## 5  4.00  4.62  3.11  4.78  4.63
## 6  7.78  5.64  3.16  5.36  5.30
```

```r
# Sort by c3(descending) and c4(acending)
df <-data_frame[order(-data_frame$c3, data_frame$c4),]
head(df)
```

```
## # A tibble: 6 x 5
##      c1    c2    c3    c4    c5
##   <dbl> <dbl> <dbl> <dbl> <dbl>
## 1  5.77  4.63  8.36  4.64  5.59
## 2  4.70  5.27  7.67  3.90  3.58
## 3  6.63  4.44  7.49  4.88  6.26
## 4  2.63  4.27  7.11  4.58  6.12
## 5  5.04  5.83  6.97  7.05  5.74
## 6  4.68  7.22  6.76  4.87  6.55
```

