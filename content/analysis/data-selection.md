---
title: Data Selection
author: Thomas
date: []
slug: data-selection
categories: []
tags:
  - preparation
header:
  caption: ''
  image: ''
---

## Data selection

In this chapter, you will learn how to manipulate the data. The library called `dplyr` contains valuable verbs to navigate inside the dataset.

Through this chapter, you will use the `Travel times` dataset. The dataset collects information on the trip leads by a driver between his home and his workplace. There are fourteen variables in the dataset, including:

- `DayOfWeek`: Identify the day of the week the driver uses his car
- `Distance`: The total distance of the journey
- `MaxSpeed`: The maximum speed of the journey
- `TotalTime`: The length in minutes of the journey

The dataset has around 200 observations in the dataset, and the rides occurred between Monday to Friday.

First of all, you need to:

- load the dataset
- check the structure of the data. 

One handy feature with `dplyr` is the `glimpse()` function. This is an improvement over `str()`. We can use `glimpse()` to see the structure of the dataset and decide what manipulation is required.


```r
library(dplyr)
PATH <- "https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/travel_times.csv"
df <- read.csv(PATH)
glimpse(df)
```

```
## Observations: 205
## Variables: 14
## $ X              <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, ...
## $ Date           <fct> 1/6/2012, 1/6/2012, 1/4/2012, 1/4/2012, 1/3/201...
## $ StartTime      <fct> 16:37, 08:20, 16:17, 07:53, 18:57, 07:57, 17:31...
## $ DayOfWeek      <fct> Friday, Friday, Wednesday, Wednesday, Tuesday, ...
## $ GoingTo        <fct> Home, GSK, Home, GSK, Home, GSK, Home, GSK, GSK...
## $ Distance       <dbl> 51.29, 51.63, 51.27, 49.17, 51.15, 51.80, 51.37...
## $ MaxSpeed       <dbl> 127.4, 130.3, 127.4, 132.3, 136.2, 135.8, 123.2...
## $ AvgSpeed       <dbl> 78.3, 81.8, 82.0, 74.2, 83.4, 84.5, 82.9, 77.5,...
## $ AvgMovingSpeed <dbl> 84.8, 88.9, 85.8, 82.9, 88.1, 88.8, 87.3, 85.9,...
## $ FuelEconomy    <fct> , , , , , , -, -, 8.89, 8.89, 8.89, 8.89, 8.89,...
## $ TotalTime      <dbl> 39.3, 37.9, 37.5, 39.8, 36.8, 36.8, 37.2, 37.9,...
## $ MovingTime     <dbl> 36.3, 34.9, 35.9, 35.6, 34.8, 35.0, 35.3, 34.3,...
## $ Take407All     <fct> No, No, No, No, No, No, No, No, No, No, No, No,...
## $ Comments       <fct> , , , , , , , , , , , , , , , Put snow tires on...
```

This is obvious that the variable `Comments` needs further diagnostic. The first observations of the `Comments` variable are only missing values. 


```r
sum(df$Comments =="")
```

```
## [1] 181
```

Code Explanation

- `sum(df$Comments =="")`: Sum the observations equalts to `""` in the column `comments` from `df`

### Filter observations from statement

**Function `select()`**

We will begin with the `select()` verb. We don't necessarily need all the variables, and a good practice is to select only the variables you find relevant.

We have 181 missing observations, almost 90 percent of the dataset. If you decide to exclude them, you won't be able to carry on the analysis. 

The other possibility is to drop the variable `Comment` with the `select()` verb.

We can select variables in different ways with `select()`. Note that, the first argument is the dataset.

```
- `select(df, A, B ,C)`: Select the variables A, B and C from df dataset.
- `select(df, A:C)`: Select all variables from A to C from df dataset.
- `select(df, -C)`: Exclude C from the dataset from df dataset.
```

You can use the third way to exclude the `Comments` variable. 


```r
step_1_df <- select(df, -Comments)
dim(df)
```

```
## [1] 205  14
```

```r
dim(step_1_df)
```

```
## [1] 205  13
```

The original dataset has 14 features while the `step_1_df` has 13.

**Function `Filter()`**

The `filter()` verb helps to keep the observations following a criteria. The `filter()` works exactly like `select()`, you pass the data frame first and then a condition separated by a comma:

```
filter(df, condition)
arguments:

- df: dataset used to filter the data
- condition:  Condition used to filter the data
```

**One criteria**

First of all, you can count the number of observations within each level of a factor variable.


```r
table(step_1_df$GoingTo)
```

```
## 
##  GSK Home 
##  105  100
```

Code Explanation

- `table()`: Count the number of observations by level. Note, only factor level variable are accepted
- `table(step_1_df$GoingTo)`: Count the number of of trips toward the final destination.

The function `table()` indicates 105 rides are going to `GSK` and 100 to `Home`.

We can filter the data to return one dataset with 105 observations and another one with 100 observations.


```r
# Select observations if GoingTo == Home
select_home <- filter(df, GoingTo =="Home")
dim(select_home)
```

```
## [1] 100  14
```

```r
# Select observations if GoingTo == Work
select_work <- filter(df, GoingTo =="GSK")
dim(select_work)
```

```
## [1] 105  14
```

**Multiple criterions**

We can filter a dataset with more than one criteria. For instance, you can extract the observations where the destination is `Home` and occured on a `Wednesday`.


```r
select_home_wed <- filter(df, GoingTo =="Home" & DayOfWeek == "Wednesday")
dim(select_home_wed)
```

```
## [1] 23 14
```

23 observations matched this criterion.

### Pipeline

The creation of a dataset requires a lot of operations, such as:

- importing
- merging
- selecting 
- filtering 
- and so on 

The `dplyr` library comes with a practical operator, `%>%`, called the **pipeline**. The pipeline feature makes the manipulation clean, fast and less prompt to error. 

This operator is a code which performs steps without saving intermediate steps to the hard drive. 
If you are back to our example from above, you can select the variables of interest and filter them. We have three steps:

- Step 1: Import data: Import the gps data
- Step 2: Select data: Select `GoingTo` and `DayOfWeek`
- Step 3: Filter data: Return only `Home` and `Wednesday`

We can use the hard way to do it:


```r
# Step 1
step_1 <- read.csv(PATH)
# Step 2
step_2 <- select(step_1, GoingTo, DayOfWeek)
# Step 3
step_3 <- filter(step_2, GoingTo=="Home", DayOfWeek=="Wednesday")
head(step_3)
```

```
##   GoingTo DayOfWeek
## 1    Home Wednesday
## 2    Home Wednesday
## 3    Home Wednesday
## 4    Home Wednesday
## 5    Home Wednesday
## 6    Home Wednesday
```

That is not a convenient way to perform many operations, especially in a situation with lots of steps. The environment ends up with a lot of objects stored.

Let's use the pipeline operator `%>%` instead. We only need to define the data frame used at the beginning and all the process will flow from it. 

Basic syntax of pipeline

```
New_df <- df %>%
                step 1 %>%
                step 2 %>%
                ...
arguments

- New_df: Name of the new data frame 
- df: Data frame used to compute the step
- step: Instruction for each step
- Note: The last instruction does not need the pipe operator `%`, you don't have instructions to pipe anymore

Note: Create a new variable is optional. If not included, the output will be displayed in the console.
```

You can create your first pipe following the steps enumerated above.


```r
# Create the data frame filter_home_wed. It will be the object return at the end of the pipeline
filter_home_wed <- 
  # Step 1
  read.csv(PATH) %>%
  # Step 2
                   select(GoingTo, DayOfWeek) %>%
  # Step 3
                   filter(GoingTo=="Home", DayOfWeek=="Wednesday")

identical(step_3, filter_home_wed)
```

```
## [1] TRUE
```

We are ready to create a stunning dataset with the pipeline operator.

**Function `arrange()`**

In the previous tutorial, you learn how to sort the values with the function `sort()`. The library `dplyr` has its sorting function. It works like a charm with the pipeline. The `arrange()` verb can reorder one or many rows, either ascending (default) or descending. 

```
- `arrange(A)`: Ascending sort of variable A
- `arrange(A, B)`: Ascending sort of variable A and B
- `arrange(desc(A), B)`: Descending sort of variable A and ascending sort of B
```

We can sort the distance by destination. 


```r
# Sort by destination and distance
step_2_df <- step_1_df %>%
             arrange(GoingTo, Distance)
head(step_2_df)
```

```
##     X       Date StartTime DayOfWeek GoingTo Distance MaxSpeed AvgSpeed
## 1 193  7/25/2011     08:06    Monday     GSK    48.32    121.2     63.4
## 2 196  7/21/2011     07:59  Thursday     GSK    48.35    129.3     81.5
## 3 198  7/20/2011     08:24 Wednesday     GSK    48.50    125.8     75.7
## 4 189  7/27/2011     08:15 Wednesday     GSK    48.82    124.5     70.4
## 5  95 10/11/2011     08:25   Tuesday     GSK    48.94    130.8     85.7
## 6 171  8/10/2011     08:13 Wednesday     GSK    48.98    124.8     72.8
##   AvgMovingSpeed FuelEconomy TotalTime MovingTime Take407All
## 1           78.4        8.45      45.7       37.0         No
## 2           89.0        8.28      35.6       32.6        Yes
## 3           87.3        7.89      38.5       33.3        Yes
## 4           77.8        8.45      41.6       37.6         No
## 5           93.2        7.81      34.3       31.5        Yes
## 6           78.8        8.54      40.4       37.3         No
```

### Summary

In the table below, you summarize all the operations you learnt during the tutorial.

| Verb      | Objective                                    | Code                               | Explanation                                           |
|-----------|----------------------------------------------|------------------------------------|-------------------------------------------------------|
| glimpse   | check the structure of a df                  | glimpse(df)                        | Identical to str()                                    |
| select()  | Select/exclude the variables                 | select(df, A, B ,C)                | Select the variables A, B and C                       |
|           |                                              | select(df, A:C)                    | Select all variables from A to C                      |
|           |                                              | select(df, -C)                     | Exclude C                                             |
| filter()  | Filter the df based a one or many conditions | filter(df, condition1)             | One condition                                         |
|           |                                              | filter(df, condition1 | ondition2) | One condition with OR operator                        |
| arrange() | Sort the dataset with one or many variables  | arrange(A)                         | Ascending sort of variable A                          |
|           |                                              | arrange(A, B)                      | Ascending sort of variable A and B                    |
|           |                                              | arrange(desc(A), B)                | Descending sort of variable A and ascending sort of B |
| %>%       | Create a pipeline between each step          | step 1 %>% step 2 %>% step 3       |                                                       |
### Aggregate the dataset

Summary of a variable is important to have an idea about the data. Although, summarizing a variable by group gives better information on the distribution of the data.

In this tutorial, you will learn how summarize a dataset by group with the `dplyr` library.

For this tutorial, you will use the `batting` dataset. The original dataset contains 102816 observations and 22 variables. You will only use 20 percent of this dataset and use the following variables:

- `playerID`: Player ID code. Factor
- `yearID`: Year. Factor
- `teamID`: Team. factor
- `lgID`: League. Factor: AA AL FL NL PL UA
- `AB`:  At bats. Numeric
- `G`: Games: number of games by a player. Numeric
- `R`: Runs. Numeric
- `HR`: Homeruns. Numeric
- `SH`: Sacrifice hits. Numeric

Before you perform summary, you will do the following steps to prepare the data:

- Step 1: Import the data
- Step 2: Select the relevant variables
- Step 3: Sort the data


```r
library(dplyr)
# Step 1
data <-  read.csv("https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/batting_lahman.csv")  %>%
# Step 2
  select(c(playerID, yearID, AB, teamID, lgID, G, R, HR, SH)) %>%
# Step 3  
  arrange(playerID, teamID, yearID)
```

A good practice when you import a dataset is to use the `glimpse()` function to have an idea about the structure of the dataset. 


```r
# Structure of the data
glimpse(data)
```

```
## Observations: 20,563
## Variables: 9
## $ playerID <fct> aardsda01, aardsda01, aaronha01, aaronha01, aaronha01...
## $ yearID   <int> 2009, 2010, 1973, 1957, 1962, 1975, 1986, 1979, 1980,...
## $ AB       <int> 0, 0, 392, 615, 592, 465, 0, 0, 0, 0, 0, 0, 45, 610, ...
## $ teamID   <fct> SEA, SEA, ATL, ML1, ML1, ML4, BAL, CAL, CAL, CAL, LAN...
## $ lgID     <fct> AL, AL, NL, NL, NL, AL, AL, AL, AL, AL, NL, AL, NA, N...
## $ G        <int> 73, 53, 120, 151, 156, 137, 66, 37, 40, 24, 32, 18, 1...
## $ R        <int> 0, 0, 84, 118, 127, 45, 0, 0, 0, 0, 0, 0, 3, 70, 20, ...
## $ HR       <int> 0, 0, 40, 44, 45, 12, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0...
## $ SH       <int> 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, NA, 5, 8, 0, NA, ...
```

**Function `summarise()`**

The syntax of `summarise()` is basic and consistent with the other verbs included in the `dplyr` library.

```
summarise(df, variable_name = condition)
arguments:

- `df`: Dataset used to construct the summary statistics
- `variable_name = condition`: Formula to create the new variable
```

Look at the code below:


```r
summarise(data, mean_run = mean(R))
```

```
##   mean_run
## 1 19.20114
```

Code Explanation

- `summarise(data, mean_run = mean(R))`: Creates a variable named `mean_run` which is the average of the column `run` from the dataset `data`. 

You can add as many variables as you want. You return the average games played and the average sacrifice hits.  


```r
summarise(data, mean_games = mean(G),
                mean_SH = mean(SH, na.rm = TRUE))
```

```
##   mean_games  mean_SH
## 1   51.98361 2.340085
```

Code Explanation

- `mean_SH = mean(SH, na.rm = TRUE)`: Summarize a second variable. You set `na.rm = TRUE` because the column `SH` contains missing observations.

### Group_by vs no group_by

The function `summerise()` without `group_by()` does not make any sense. It is more instructive to construct a summary statistic by group. The library `dplyr` applies a function automatically to the group you passed inside the verb `group_by`. 

Note that, `group_by` works perfectly with all the other verbs (i.e. `mutate()`, `filter()`, `arrange()`, ...). 

It is convenient to use the pipeline operator when you have more than one step. You can compute the average homerun by baseball league.


```r
data %>%
  group_by(lgID) %>%
  summarise(mean_run = mean(HR))
```

```
## # A tibble: 7 x 2
##   lgID  mean_run
##   <fct>    <dbl>
## 1 AA       0.917
## 2 AL       3.13 
## 3 FL       1.31 
## 4 NL       2.86 
## 5 PL       2.58 
## 6 UA       0.622
## 7 <NA>     0.287
```

Code Explanation

- `data`: Dataset used to construct the summary statistics
- `group_by(lgID)`: Compute the summary by grouping the variable `lgID
- `summarise(mean_run = mean(HR))`: Compute the average homerun

The pipe operator works with `ggplot()` as well. You can easily show the summary statistic with a graph. All the steps are pushed inside the pipeline until the grap is plot. It seems more visual to see the average homerun by league with a bar char. The code below demonstrates the power of combining `group_by()`, `summarise()` and `ggplot()` together.

You will do the following step:

- Step 1: Select data frame
- Step 2: Group data
- Step 3: Summarize the data
- Step 4: Plot the summary statistics


```r
library(ggplot2)

# Step 1
data %>% 
# Step 2  
  group_by(lgID)%>%
# Step 3  
  summarise(mean_home_run = mean(HR)) %>%
# Step 4  
  ggplot(aes(x = lgID, y = mean_home_run, fill = lgID))+
  geom_bar(stat="identity") +
  theme_classic()+ 
  labs(
    x ="baseball league",
    y = "Average home run",
    title = paste(
        "Example group_by() with summarise()"
    )
  )
```

![](/project/data-selection_files/example1-1.png)<!-- -->

### Functions compatible 

The verb `summarise()` is compatible with almost all the functions in R. Here is a short list of useful functions you can use together with `summarise()`:

| Objective | Function     | Description                                                   |
|-----------|--------------|---------------------------------------------------------------|
| Basic     | mean()       | Average of vector x                                           |
|           | median()     | Median of vector x                                            |
|           | sum()        | Sum of vector x                                               |
| variation | sd()         | standard deviation of vector x                                |
|           | IQR()        | Interquartile of vector x                                     |
| Range     | min()        | Minimum of vector x                                           |
|           | max()        | Maximum of vector x                                           |
|           | quantile()   | Quantile of vector x                                          |
| Position  | first()      | Use with group_by() First observation of the group                |
|           | last()       | Use with group_by(). Last observation of the group            |
|           | nth()        | Use with group_by(). nth observation of the group             |
| Count     | n()          | Use with group_by(). Count the number of rows                 |
|           | n_distinct() | Use with group_by(). Count the number of distinct observations |

In the previous example, you didn't store the summary statistic in a data frame.

You can proceed in two steps to generate a date frame from a summary:

- Step 1: Store the data frame for further use
- Step 2: Use the dataset to create a line plot

**Step 1**

You compute the average number of games played by year.


```r
## Mean
ex1 <- data %>%
    group_by(yearID) %>%
    summarise(mean_game_year = mean(G)) 
head(ex1)
```

```
## # A tibble: 6 x 2
##   yearID mean_game_year
##    <int>          <dbl>
## 1   1871           23.4
## 2   1872           18.4
## 3   1873           25.6
## 4   1874           39.1
## 5   1875           28.4
## 6   1876           35.9
```

Code Explanation

- The summary statistic of `batting` dataset is stored in the data frame `ex1`. 

**Step 2**

you show the summary statistic with a line plot and see the trend.


```r
# Plot the graph
ggplot(ex1, aes(x = yearID, y = mean_game_year)) +
  geom_line() + 
  theme_classic() +
    labs(
    x ="Year",
    y = "Average games played",
    title = paste(
        "Average games played from 1871 to 2016"
    )
  )
```
<img src="/project/data-selection_files/ex2_plot-1.png" width="55%" style="display: block; margin: auto;" />

### Subsetting

The function `summarise()` is compatible with subsetting. 


```r
## Subsetting + Median
data %>%
    group_by(lgID) %>%
    summarise(median_at_bat_league = median(AB), 
              # Compute the median without the zero
              median_at_bat_league_no_zero = median(AB[AB > 0])) 
```

```
## # A tibble: 7 x 3
##   lgID  median_at_bat_league median_at_bat_league_no_zero
##   <fct>                <dbl>                        <dbl>
## 1 AA                    130.                         131.
## 2 AL                     38.                          85.
## 3 FL                     88.                          97.
## 4 NL                     56.                          67.
## 5 PL                    238.                         238.
## 6 UA                     35.                          35.
## 7 <NA>                  101.                         101.
```

Code Explanation

- `median_at_bat_league_no_zero = median(AB[AB > 0])`: The variable `AB` contains lots of 0. You can compare the median of the **at bat** variable with and without 0.

**Sum**

Another useful function to aggregate the variable is `sum()`. 

You can check which leagues have the more homeruns. 


```r
## Sum
data %>%
    group_by(lgID) %>%
    summarise(sum_homerun_league = sum(HR)) 
```

```
## # A tibble: 7 x 2
##   lgID  sum_homerun_league
##   <fct>              <int>
## 1 AA                   341
## 2 AL                 29426
## 3 FL                   130
## 4 NL                 29817
## 5 PL                    98
## 6 UA                    46
## 7 <NA>                  41
```

**Standard deviation**

Spread in the data is computed with the standard deviation or `sd()` in R.  


```r
# Spread
data %>%
    group_by(teamID) %>%
    summarise(sd_at_bat_league = sd(HR)) 
```

```
## # A tibble: 148 x 2
##    teamID sd_at_bat_league
##    <fct>             <dbl>
##  1 ALT              NA    
##  2 ANA               8.78 
##  3 ARI               6.08 
##  4 ATL               8.54 
##  5 BAL               7.74 
##  6 BFN               1.36 
##  7 BFP               0.447
##  8 BL1               0.699
##  9 BL2               1.71 
## 10 BL3               1.00 
## # ... with 138 more rows
```

There are lots of inequality in the quantity of homerun done by each team.

**Minimum and maximum**

You can access the minimum and the maximum of a vector with the function `min()` and `max()`. 

The code below returns the lowest and highest number of games in a season played by a player. 


```r
# Min and max
data %>%
    group_by(playerID) %>%
    summarise(min_G = min(G),
              max_G = max(G)) 
```

```
## # A tibble: 10,395 x 3
##    playerID  min_G max_G
##    <fct>     <dbl> <dbl>
##  1 aardsda01   53.   73.
##  2 aaronha01  120.  156.
##  3 aasedo01    24.   66.
##  4 abadfe01    18.   18.
##  5 abadijo01   11.   11.
##  6 abbated01    3.  153.
##  7 abbeybe01   11.   11.
##  8 abbeych01   80.  132.
##  9 abbotgl01    5.   23.
## 10 abbotji01   13.   29.
## # ... with 10,385 more rows
```

**Count**

Count observations by group is always a good idea. With R, you can aggregate the the number of occurence with `n()`. 

For instance, the code below computes the number of years played by each player.


```r
# count observations
data %>%
    group_by(playerID) %>%
    summarise(number_year = n()) %>%
    arrange(desc(number_year))
```

```
## # A tibble: 10,395 x 2
##    playerID  number_year
##    <fct>           <int>
##  1 pennohe01          11
##  2 joosted01          10
##  3 mcguide01          10
##  4 rosepe01           10
##  5 davisha01           9
##  6 johnssi01           9
##  7 kaatji01            9
##  8 keelewi01           9
##  9 marshmi01           9
## 10 quirkja01           9
## # ... with 10,385 more rows
```

**First and last**

You can select the first, last or nth position of a group. 

For instance, you can find the first and last year of each player.  


```r
# first and last
data %>%
      group_by(playerID) %>%
      summarise(first_appearance = first(yearID),
                last_appearance = last(yearID))
```

```
## # A tibble: 10,395 x 3
##    playerID  first_appearance last_appearance
##    <fct>                <int>           <int>
##  1 aardsda01             2009            2010
##  2 aaronha01             1973            1975
##  3 aasedo01              1986            1990
##  4 abadfe01              2016            2016
##  5 abadijo01             1875            1875
##  6 abbated01             1905            1897
##  7 abbeybe01             1894            1894
##  8 abbeych01             1895            1897
##  9 abbotgl01             1973            1979
## 10 abbotji01             1992            1996
## # ... with 10,385 more rows
```

**nth observation**

The fonction `nth()` is complementary to `first()` and `last()`. 
You can access the nth observation within a group with the index to return. 

For instance, you can filter only the second year that a team played.  


```r
# nth
data %>%
      group_by(teamID) %>%
      summarise(second_game = nth(yearID,2)) %>%
      arrange(second_game)
```

```
## # A tibble: 148 x 2
##    teamID second_game
##    <fct>        <int>
##  1 BS1           1871
##  2 CH1           1871
##  3 FW1           1871
##  4 NY2           1871
##  5 RC1           1871
##  6 BR1           1872
##  7 BR2           1872
##  8 CL1           1872
##  9 MID           1872
## 10 TRO           1872
## # ... with 138 more rows
```

**Distinct number of observations**

The function `n()` returns the number of observations in a current group. A closed function to `n()` is `n_distinct()`, which count the number of unique values. 

In the next example, you add up the total of players a team recruited during the all periods. 


```r
# distinct values
data %>%
    group_by(teamID) %>%
    summarise(number_player = n_distinct(playerID)) %>%
    arrange(desc(number_player))
```

```
## # A tibble: 148 x 2
##    teamID number_player
##    <fct>          <int>
##  1 CHN              751
##  2 SLN              729
##  3 PHI              699
##  4 PIT              683
##  5 CIN              679
##  6 BOS              647
##  7 CLE              646
##  8 CHA              636
##  9 DET              623
## 10 NYA              612
## # ... with 138 more rows
```

Code Explanation

-	`group_by(teamID)`: Group by year and team
		`summarise(number_player = n_distinct(playerID))`: Count the distinct number of players by team
		`arrange(desc(number_player))`: Sort the data by the number of player

### Multiple groups

A summary statistic can be realized among multiple groups. 


```r
# Multiple groups
data %>%
  group_by(yearID, teamID) %>%
  summarise(mean_games = mean(G)) %>%
  arrange(desc(teamID, yearID))
```

```
## # A tibble: 2,829 x 3
## # Groups:   yearID [146]
##    yearID teamID mean_games
##     <int> <fct>       <dbl>
##  1   1884 WSU         20.4 
##  2   1891 WS9         46.3 
##  3   1886 WS8         22.0 
##  4   1887 WS8         51.0 
##  5   1888 WS8         27.0 
##  6   1889 WS8         52.4 
##  7   1884 WS7          8.00
##  8   1875 WS6         14.8 
##  9   1873 WS5         16.6 
## 10   1872 WS4          4.20
## # ... with 2,819 more rows
```

Code Explanation

- `group_by(yearID, teamID)`:  Group by year **and**  team
- `summarise(mean_games = mean(G))`:  Summarize the number of game player
- `arrange(desc(teamID, yearID))`: Sort the data by team and year

### Filter

Before you intend to do an operation, you can filter the dataset. The dataset starts in 1871, and the analysis does not need the years prior to 1980.


```r
# Filter
data %>%
    filter(yearID > 1980) %>%
    group_by(yearID) %>%
    summarise(mean_game_year = mean(G)) 
```

```
## # A tibble: 36 x 2
##    yearID mean_game_year
##     <int>          <dbl>
##  1   1981           40.6
##  2   1982           57.0
##  3   1983           60.3
##  4   1984           63.0
##  5   1985           57.8
##  6   1986           58.6
##  7   1987           48.7
##  8   1988           52.6
##  9   1989           58.2
## 10   1990           52.9
## # ... with 26 more rows
```

Code Explanation

- `filter(yearID > 1980)`: Filter the data to show only the relevant years (i.e. after 1980)
- `group_by(yearID)`: Group by year
- `summarise(mean_game_year = mean(G))`: Summarize the data

### Ungroup 

Last but not least, you need to remove the grouping before you want to change the level of the computation. 


```r
# Ungroup the data
data %>%
     filter(HR >0)%>%
     group_by(playerID) %>%
     summarise(average_HR_game = sum(HR)/sum(G)) %>%
     ungroup() %>%
     summarise(total_average_homerun = mean(average_HR_game))
```

```
## # A tibble: 1 x 1
##   total_average_homerun
##                   <dbl>
## 1                0.0688
```

Code Explanation

- `filter(HR >0)` :  Exclude zero homerun
- `group_by(playerID)`:  group by player
- `summarise(average_HR_game = sum(HR)/sum(G))`: Compute average homerun by player
- `ungroup()`:  remove the grouping
- `summarise(total_average_homerun = mean(average_HR_game))`: Summarize the data

### Summary

When you want to return a summary by group, you can use:

```
# group by X1, X2, X3
group(df, X1, X2, X3)
```
you need to ungroup the data with:

```
ungroup(df)
```

The table below summarizes the function you learnt with `summarise()`

| method                        | function   | code                                          |
|-------------------------------|------------|-----------------------------------------------|
| mean                          | mean       | summarise(df,mean_x1 = mean(x1))             |
| median                        | median     | summarise(df,median_x1 = median(x1))         |
| sum                           | sum        | summarise(df,sum_x1 = sum(x1))               |
| standard deviation            | sd         | summarise(df,sd_x1 = sd(x1))                 |
| interquartile                 | IQR        | summarise(df,interquartile_x1 = IQR(x1))     |
| minimum                       | min        | summarise(df,minimum_x1 = min(x1))           |
| maximum                       | max        | summarise(df,maximum_x1 = max(x1))           |
| quantile                      | quantile   | summarise(df,quantile_x1 = quantile(x1))     |
| first observation             | first      | summarise(df,first_x1 = first(x1))           |
| last observation              | last       | summarise(df,last_x1 = last(x1))             |
| nth observation               | nth        | summarise(df,nth_x1 = nth(x1, 2))            |
| number of occurrence          | n          | summarise(df,n_x1 = n(x1))                   |
| number of distinct occurrence | n_distinct | summarise(df,n_distinct _x1 = n_distinct(x1)) |
