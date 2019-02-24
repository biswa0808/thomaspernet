---
title: Data analysis for econometrics
author: thomas
date: '2019-02-24'
slug: []
categories: []
tags: []
header:
  caption: ''
  image: ''

---

## Equation

This library proposes to define a very simple way to describe data from an equation. The explanatory data analysis is one of the first step to perform before proceeding to the modeling part. A strong and rigorous data analysis helps to anticipate and by some extent highlights potential issues that the model can face during the estimation. Besides, during the EDA, some unexplored important features can be detected, like non-linearity or outliers. 

The researcher is equipped with a vast toolkit to explore his dataset. In this library, we aim at rationalize the EDA using standardize methods and providing a quick analysis. The main idea is to provide a tool to avoid coding all over again the basic charts and statistical test required to get the best insight from the dataset. Once the researcher got his first insights within minutes, he can really focus on how to bring more value added to the model to estimate. 

The starting point of this library is an econometrics equation to estimate. From this equation, the library will adjust the data following statistical methods and provides different charts to emerge key facts. 

An equation has the following form:

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/0.png" >

The goal is to analyze the variance of the independent variables on Y. To extract major insights, we distinguish two types of categorical variable. We call the first type `i`,  `low dimensional` which gather less than five unique groups. By analogy, the second type is `high dimensional` groups. The second group is highly important to analyze because of the large heterogeneity between unique groups, so dramatically affect the estimation if not treated carefully.

An equation can be composed by either a `one-way`, a `two-way` or both fixed effect. Say differently, an equation with only a `one-way` fixed effect estimates the coefficient of a dummy for `n-1` group, while a `two-way` fixed effect estimates `n-1` coefficient for a pair of categorical variables. In the later group, a precise analysis needs to be done to understand the structure of the pair. Indeed, a pair with only a single observation does not add any useful information to the model, instead, it affects the computation of both the coefficient and the standard error (ie. matrix inverse is not properly calculated). 

Last but not least, an equation is often estimated with a time variable. 

## How to use 

The library is composed by four modules:

- `preprocessing`: Prepare the dataset
- `categorical`: visualize categorical data
- `continuous: visualize continuous data
- `mixe`:  visualize mixes of continuous and categorical data

The researcher needs to define an equation on its own, with the first value is the independent variables while the following values are the dependent variables. The orders from the RHS variables does not matter. So far, the library works only when the Y is a continuous variables and the RHS variables cannot be quadratic, interacted within each other. Besides, the library does not take care of the outliers yet!

The equation should use the raw data, namely without transformation. The preprocessing module takes care of the variables that are not continuously distributed. It uses the log normal distribution if the Shapiro test failed to reject H0. 

Once the equation is defined, the library will project four different analysis for each type of data:

- Time series
- Categorical
- Mix
- Fixed effect

Note that, the library does not shows one way fixed effect and does not show graph between fixed effect and the dependent variable yet. 

A jupyter notebook is available [here](https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/Example_1.ipynb) with an example. 

## Example

Let's see in action an example of the analysis done by the library with the Chinese export data. 

### Dataset

The dataset is composed by six variables:

- `Sum_value`: Sum of the exported value by a Chinese city and type of trade
- `sum_quantity`: Sum of the quantity exported by a Chinese city and type of trade
- `Trade_type`: two groups:  
    - processing trader : 加工贸易
    - ordinary traders:  一般贸易
- `city_prod`: up to 250 Chinese city
- `HS_3`:  Up to 100 industries
- `Date: Trade data from 2000 to 2010

The dataset has more than 400k observations. 

The purposes is to estimate the following equation:

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/0_1.png" >

The types of data are distributed as follow:


| Variables    | continuous | low dimensional | high dimensional | Date |
| ------------ | ----------: | --------------- | ---------------- | ---- |
| sum_value    | Yes        |       -          |          -        |  -    |
| Trade_type   |      -      | Yes             |            -      |    -  |
| city_prod    |       -     |    -             | Yes                |    -  |
| HS_3         |        -    |   -              | Yes                |   -   |
| sum_quantity | Yes          |   -              |           -       |   -   |
| Date         |         -   | -                |             -     | Yes    |

### Preprocessing

The first step requires to define the equation as a list, and determine the time series variable if any. 


The reconstructed dataset,

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/1.png" >

A dictionary with the new equation to estimate and the type of data

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/2.png" >

The preprocessing steps generates the list of all possibles combination between the categorical variables and continuous variables for each different kinds of plots and tests. The researcher can use a list comprehension to iterate over all possibles choices for each group of analysis. Some example are explicitly provided in the Jupyter notebook.

### Time serie plot

The first group of graphs displays the evolution of the dependent variable with the time series variable and shows the box plot for different years. It uses the `mix` module.

**Evolution of sum of value with year**

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/3.png" >

**Distribution of sum of value with year**

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/4.png" >

### Categorical analysis

The second analysis plots the count of observation for each group. It evaluates the dependencies with all the low dimension and high dimension groups using the chi test. 

**Count of observation**

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/5.png" >

**Dependency test**

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/6.png" >

### Mix analysis

The third group of graphs mixes the different low dimensional variables with the continuous variables, including the dependent variable.

The first bunch of graphs shows the distribution the the `sum_value` and each low dimensional variable. The third graph demonstrates the empirical cumulated distribution while the fourth graph displays the coefficient of variation for each group

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/7.png" >

Next, the scatterplot matrix highlights the relationship between all the continuous variables.

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/8.png" >

Last, we can see if each continuous variable are distributed differently by low dimensional group against the dependent variable.

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/9.png" >

### Focus on fixed effect

At last, the analysis ends up by showing different perspective of the fixed effect. Fixed effect are important, they can dramatically change the sign or the standard error of an estimate while not properly constructed. The idea behind the last part of the analysis is to provide a tool to visualize some potential issues emerging from the high dimensional group. 

**Average `sum_value` by high dimensional group**

To begin with, we plot an heatmap of the average values of the dependent variable for each high dimensional variable. 

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/10.png" ><img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/11.png" >

**Average log transformed dependent variable by high dimensial group**

The next graph provides a different view of the relationship between the **transformed** dependent variable and the high dimensional variable. the function sort the average values of `sum_value_log` for each high dimensional variables and plot them.

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/12.png" ><img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/13.png" >

**Rank dependent variable with low/high dimensional**

The next plot displays a slightly different way to understand the relationship between the dependent variable and the high/low dimensional groups. 

First of all, we computed the average `sum_value_log` by `city_prod` and `Trade_type`. We, then, rank them accordingly. Finally, we plot them together to see if there is a different structure between the low/high dimension. 

We want to see if the best cities,  ie high dimensional, in terms of value exported, (dependent variable), remains the best for each trade type, ie low dimensional.

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/14.png" ><img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/15.png" >

**Count of observation by fixed effect**

The last plot shows the count of observation for each pair of fixed effect, namely two-ways. The plot helps to visualize where the most observations are concentrated. A darker red value shows a strong number of observations while gray values emphasize only one count. We need to focus on those single observation since there are likely to affect the final estimate. The blue values indicates no observation.

<img src="https://github.com/thomaspernet/Data_analysis_econometrics/raw/master/img/16.png" >