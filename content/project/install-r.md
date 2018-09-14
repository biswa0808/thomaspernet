---
title: Install R
author: Thomas
date: '2018-08-11'
slug: install-r
categories: []
tags:
  - intro
header:
  caption: 'bubbles.jpg'
  image: 'headers/bubbles-wide.jpg'
#image_preview: 'R.jpeg'  
---

R is a programming language. To use R, we need to install an **Integrated Development Environment** (IDE). **Rstudio** is the Best IDE available as it is user-friendly, open-source and is part of the Anaconda platform.

## Install Anaconda

What is Anaconda?

Anaconda free open source distributing both Python and R programming language. Anaconda is widely used in the scientific community and data scientist to carry out Machine Learning project or data analysis. 

Why use Anaconda?

Anaconda will help you to manage all the libraries required for Python or R. For the sake of the tutorial; we will use Rstudio, which is a free and open-source IDE. Rstudio is also available from direct downloading. I recommend you to install Rstudio with Anaconda especially if you are going to use or using Python. Anaconda will install all the required libraries and IDE into one single folder to simplify package management. 

For the purpose of this textbook, we will set an R environment within Anaconda. This environment will contain only the library we need. The advantage is that no library will be in conflict with the main and/or other R environment.;

To use Rstudio with Anaconda, you need to follow these steps:

- Step 1: Install Anaconda
- Step 2: Install R and Rstudio

Go to [Anaconda](https://www.anaconda.com/download/) and Download Anaconda for Python 3.6 for your OS.

By default, Chrome selects the downloading page of your system. In this tutorial installation is done for Mac. If you run on Windows or Linux, download Anaconda 5.1 for Windows installer or Anaconda 5.1 for Linux installer.

When you are done installing Anaconda, you can proceed to the installation of R and Rstudio. You will use the terminal to perform these two steps.

Open the terminal

**Mac user**

For Mac user, there is two options top open the terminal.

-	The shortest way is to use the Spotlight Search and write terminal.
   The second option is to use the shortcut `shift + command + U` and open the terminal.

<img src="/project/install-r_files/0_1.png" width="55%" style="display: block; margin: auto;" />


**Windows user**

-	Click on the "Start" button to open the Start menu.
		Open the "All Programs" menu, followed by the "Accessories" option.
		Select the "Command Prompt" option from the "Accessories" menu to open a command-line interface session in a new window on the computer.

<img src="/project/install-r_files/0_2.png" width="55%" style="display: block; margin: auto;" />

### Install R and Rstudio

When the setup is done, you are ready to install your first libraries. We recommend you to install all packages and dependencies with anaconda with the conda command in the terminal.

<img src="/project/install-r_files/0_3.png" width="55%" style="display: block; margin: auto;" />

### In the terminal

Create R environment. We will all this environment `hello-r`.

1. Set the working directory

```R
## Mac
cd anaconda3

## Windows
cd C:\Users\Admin\Anaconda3

```

2. Create a `.yml` file

```R
## Mac
touch hello-r.yml
## Windows
echo.>hello-r.yml
```

3. Edit the `.yml`file

You are ready to edit the yml file. You can paste the following code in the Terminal to edit the file. MacOS user can use vim to edit the yml file.

```R
vi hello-r.yml
```

For windows use the notepad `notepad hello-r.yml


```
name: hello-r
dependencies:
  - r-essentials
  - rstudio
  - r-randomforest
  - r-haven
  - r-caret
  - r-e1071
  - r-rocr
  - r-rcolorbrewer
  - r-bookdown
  - r-xlsx
  - r-googledrive
  - r-rdrop2
  - r-gridExtra
  - r-GGally
  - r-rpart.plot
  - r-e1071
  - r-ROCR
  - ggfortify
```

4. Create the environment

```R
conda env create -f hello-r.yml
```

it takes some time. When it's done, you can check what environment there is in Anaconda `conda env list`

5. Activate hello-r

Each time you want to use this environment, please use

**Mac OS user**

`source activate hello-r`

**Windows user**

`activate hello-r`

It takes some time to upload all the libraries. Be patient... you are all set.

If you want to see where R is located, use which r in the terminal.

<img src="/project/install-r_files/0_3_5.png" width="80%" style="display: block; margin: auto;" />

When you run Rstudio, you need to open another terminal window to write the command lines.

### Run Rstudio

Directly run the command line from terminal to open Rstudio:

<img src="/project/install-r_files/0_4.png" width="80%" style="display: block; margin: auto;" />

A new window will be opened with Rstudio.

<img src="/project/install-r_files/0_5.png" width="55%" style="display: block; margin: auto;" />

### R environment

Open Rstudio from the terminal and open a script. Write the following command:


```R
## In Rstudio
summary(cars)
```

```
##      speed           dist       
##  Min.   : 4.0   Min.   :  2.00  
##  1st Qu.:12.0   1st Qu.: 26.00  
##  Median :15.0   Median : 36.00  
##  Mean   :15.4   Mean   : 42.98  
##  3rd Qu.:19.0   3rd Qu.: 56.00  
##  Max.   :25.0   Max.   :120.00
```

```R
#2.	Click Run
#3.	Check Output
```

<img src="/project/install-r_files/0_6.png" width="55%" style="display: block; margin: auto;" />

If you can see the summary statistics, it works. You can close Rstudio without saving the files.

### Install package

Install package with anaconda is trivial. You go to your favorite browser, type the name of the library followed by anaconda r.

Note that, all the libraries from conda are already installed in the `hello-r` enviromnent. 

<img src="/project/install-r_files/0_7.png" width="55%" style="display: block; margin: auto;" />

You choose the link that points to anaconda. You copy and paste the library into the terminal.

<img src="/project/install-r_files/0_8.png" width="55%" style="display: block; margin: auto;" />

For instance, you need to install randomForest for the tutorial on random forest, you go https://anaconda.org/r/r-randomforest. 

<img src="/project/install-r_files/0_9.png" width="55%" style="display: block; margin: auto;" />

Make sure you actiavted `hello-r`e environment and then Run

```R
conda install -c r r-randomforest --yes
```
from the terminal.

<img src="/project/install-r_files/0_10.png" width="55%" style="display: block; margin: auto;" />

The installation is completed.

*Note*: Thorough this tutorial, you won't need to install many libraries as the most used libraries came with the r-essential conda library. It includes ggplot for the graph and caret for the machine learning project.

### Library oustide Conda

In some case, there are libraries not available in `conda`. You can install it using `install.packages(package_name)`. You add the path to make sure the installation goes to `hello-r` environment.

In this textbook, you need three more libraries:

- `googledrive`
- `rdrop2`
- `olsrr`

You can install them with the following code in Rstudio

```R
lib <- .libPaths()
lib
install.packages('googledrive', lib)
install.packages('rdrop2', lib)
install.packages('olsrr')
```





### Open a library

To run the R function randomForest(), we need to open the library containing the function. In the Rstudio script, we can write


```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```r
## In Rstudio
library(randomForest)
```

**Note**: Avoid as much as possible to open unnecessary packages. You might ended up creating conflicts between libraries.

### Run R code

We have two ways to run codes in R

- You can run the codes inside the Console. Your data will be stored in the Global Environment but no history is recorded. You won't be able to replicate the results once R is closed. You need to write the codes all over again. This method is not recommended if you want to replicate or save your codes.

<img src="/project/install-r_files/0_11.png" width="55%" style="display: block; margin: auto;" />

- Write the code in the script. You can write as many lines of codes as we want. To run the code, you select the rows you want to return. Finally click on run. The variable A, B and C are stored in the global environment as values and you can see the output in the Console. You can save your script and open it later. Your results won't we lost.

<img src="/project/install-r_files/0_12.png" width="55%" style="display: block; margin: auto;" />

*Note*: If you point the cursor at the third rows (i.e. `C <- A+B`), the Console displays an error. That's, you didn't run the line above C.

In this case, R does not know the value of A and B yet.

<img src="/project/install-r_files/0_13.png" width="55%" style="display: block; margin: auto;" />

In a similar way, if you point the cursor to an empty row and click on run, R return an empty output.

<img src="/project/install-r_files/0_14.png" width="55%" style="display: block; margin: auto;" />