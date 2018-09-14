---
title: Import Data
author: Thomas
date: '2018-08-11'
slug: import-data
categories: []
tags:
  - intro
header:
  caption: ''
  image: ''
---

## Import data with R

Data could exist in various formats. For each format R has a specific function and argument. This tutorial explains how to import data to R.

This chapter is divided as follow:
 
- Read CSV files
- Read Excel files
- Read SAS, SPSS, STATA files
- Read RDA  files

### Read CSV

One of the most widely data store is the `.csv` (comma-separated values) file formats. R loads an array of libraries during the start-up, including the `utils` package. This package is convenient to open csv files combined with the `reading.csv()` function. Here is the syntax for read.csv:

``` 
read.csv(file, header = TRUE, sep = ",")
argument:

-file: PATH where the file is stored
- header: confirm if the file has an header or not, by default, the header is set to TRUE
- sep: the symbol used to split the variable.By default, `,`.
```

We will read the data file name `mtcats`. The csv file is stored online. If your `.csv` file is stored locally, you can replace the PATH inside the code snippet. Don't forget to wrap it inside ' '. The PATH needs to be a string value.

For mac user, the path for the download folder is:

-  `"/Users/USERNAME/Downloads/FILENAME.csv"`

For windows user:

- `"C:\Users\USERNAME\Downloads\FILENAME.csv"`
Note that, you should always specify the extension of the file name. 

-	`.csv`
-	`.xlsx`
-	`.txt`
- ...


```r
PATH <- 'https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/mtcars.csv'

df <- read.csv(PATH, header =  TRUE, sep = ',')
length(df)
```

```
## [1] 12
```

```r
class(df$X)
```

```
## [1] "factor"
```

R, by default, returns character values as `Factor`, you can turn off this setting by adding `stringsAsFactors = FALSE`. 


```r
df <- read.csv(PATH, header =  TRUE, sep = ',', stringsAsFactors = FALSE)
class(df$X)
```

```
## [1] "character"
```

The class for the variable `X` is now a `character`.

### Read Excel files

Excel files are very popular among data analysts. Spreadsheets are easy to work with and flexible. R is equipped with a library readxl to import Excel spreadsheet.

Use this code 


```r
require(readxl) 
```

```
## Loading required package: readxl
```

to check if `readxl` is installed in your machine. If you install `r` with `r-conda-essential`, the library is already installed. You should see in the command window:

`Loading required package: readxl.`

If the package does not exit, you can install it with the conda

- `readxl`: Open excel spreadsheet. If you have install R with `r-essential`. It is already in the library
    - [Anaconda](https://anaconda.org/mittner/r-readxl): `conda install -c mittner r-readxl`

Use the following command to load the library to import excel files.


```r
library(readxl)
```

We use the examples included in the package readxl during this tutorial. 
Use code 


```r
readxl_example() 
```

```
##  [1] "clippy.xls"    "clippy.xlsx"   "datasets.xls"  "datasets.xlsx"
##  [5] "deaths.xls"    "deaths.xlsx"   "geometry.xls"  "geometry.xlsx"
##  [9] "type-me.xls"   "type-me.xlsx"
```

to see all the available spreadsheets in the libray.

To check the location of the spreadsheet named clippy.xls, simple use


```r
readxl_example("geometry.xls")
```

```
## [1] "/Users/Thomas/anaconda3/envs/hello-r/lib/R/library/readxl/extdata/geometry.xls"
```

If you install R with `conda`, the spreadsheets are located in `anaconda3/lib/R/library/readxl/extdata/geometry.xls`

The function `read_excel()` is of great use when it comes to opening xls and xlsx extention. 

The syntax is:

```
read_excel(PATH, sheet = NULL, range= NULL, col_names = TRUE)
arguments:
  
- PATH: Path where the excel is located
- sheet: Select the sheet to import. By default, all
- range: Select the range to import. By default, all non-null cells
- col_names: Select the columns to import. By default, all non-null columns
```

We can import the spreadsheets from the `readxl` library and count the number of column in the first sheet.  


```r
# Store the path of `datasets.xlsx`
example <- readxl_example("datasets.xlsx")
# Import the spreadsheet
df <- read_excel(example)
# Count the number of columns
length(df)
```

```
## [1] 5
```


The file `datasets.xlsx` is composed of 4 sheets. We can find out which sheets are available in the workbook by using `excel_sheets()` function:


```r
example <- readxl_example("datasets.xlsx")
excel_sheets(example)
```

```
## [1] "iris"     "mtcars"   "chickwts" "quakes"
```

If a worksheet includes many sheets, it is easy to select one by using the `sheet` arguments. You can specify the name of the sheet or the index. You can verify if both function returns the same output with `identical()`.


```r
quake <- read_excel(example, sheet = "quakes", col_names = TRUE)
quake_1 <-read_excel(example, sheet = 4, col_names = TRUE)
identical(quake, quake_1)
```

```
## [1] TRUE
```

You can control what cells to read in 2 ways:

1.	Use `n_max` argument to return `n` rows
2.	use `range`argument combined with `cell_rows` or `cell_cols`

You can use the argument `n_max` with the number of rows to import in R. For instance, you set `n_max` equals to 5 to import the first five rows. Note, if your spreadsheet does not have header, change `TRUE` to `FALSE` in the `col_names` argument. 


```r
# Read the first five row
iris <- read_excel(example, n_max = 5, col_names = TRUE)
```

<img src="/project/import-data_files/6.png" width="80%" style="display: block; margin: auto;" />

If you change `col_names` to `FALSE`, R creates the headers automaticaly. 


```r
# Read the first five row
iris_no_header <- read_excel(example, n_max = 5, col_names = FALSE)
```

In the data frame `iris_no_header`, R created five new variables named `X__1`, `X__2`, `X__3`, `X__4` and `X__5`

<img src="/project/import-data_files/7.png" width="80%" style="display: block; margin: auto;" />

You can also use the argument `range` to select rows and columns in the spreadsheet. In the code below, you use the excel style to select the range A1 to B5.


```r
# Read rows A1 to B5
example_1 <- read_excel(example, range = "A1:B5", col_names = TRUE)
dim(example_1)
```

```
## [1] 4 2
```

You can see that the `example_1`returns 4 rows with 2 columns. The dataset has header, that the reason the dimension is 4x2.

In the second example, you use the function `cell_rows()` which controls the range of rows to return. If you want to import the rows 1 to 5, you can set `cell_rows(1:5)`. Note that, `cell_rows(1:5)` returns the same output as `cell_rows(5:1)`.


```r
# Read 5 rows of all columns
example_2 <- read_excel(example, range = cell_rows(1:5),  col_names = TRUE)
dim(example_2)
```

```
## [1] 4 5
```

The `example_2` however is a 4x5 matrix. The iris dataset has 5 columns with header. We return the first four rows with header of all columns.
 
In case you want to import rows which do not begin in the first row, you have to include `col_names = FALSE`. If you use `range = cell_rows(2:5)`, it becomes obvious our data frame does not have header anymore. 
 

```r
iris_row_with_header <- read_excel(example, range = cell_rows(2:3), col_names = TRUE)
iris_row_no_header <- read_excel(example, range = cell_rows(2:3), col_names = FALSE)
```

<img src="/project/import-data_files/8.png" width="80%" style="display: block; margin: auto;" />

We can select the columns with the letter, like in Excel.


```r
# Select columns 1 and B
col <- read_excel(example, range = cell_cols("A:B"))
dim(col)
```

```
## [1] 150   2
```

*Note*: `range = cell_cols("A:B")`, returns output all cells with non-null value. The dataset contains 150 rows, therefore, `read_excel()` returns rows up to 150. This is verified with the `dim() function.

`read_excel()` returns `NA` when a symbol without numerical value appears in the cell. You can count the number of missing values with the combination of two functions:

1. `sum()`
2. `is.na

Here is the code


```r
iris_na <- read_excel(example, na = "setosa")
sum(is.na(iris_na))
```

```
## [1] 50
```

We have 50 values missing, which are the rows belonging to the `setosa` species.

### Import data from other Statistical software

You will import different files format with the `heaven`package. This package support SAS, STATA and SPSS softwares. We can use the folloing function to open different types of dataset, according to the extension of the file:

- SAS: `read_sas()`
- STATA: `read_dta()` (or `read_stata()`, which are identical)
- SPSS: `read_sav()` or `read_por()`. We need to check the extension

Only one argument is required within these function. You need to know the PATH where the file is stored and that's it, you are ready to open all the files from SAS, STATA and SPSS. These three function accepts an URL with the same extensions.



```r
library(haven)
```

- `haven`: Import/export data with format differetn from csv. If you have install R with `r-essential`. It is already in the library
    - [Anaconda](https://anaconda.org/conda-forge/r-haven): `conda install -c conda-forge r-haven`

**Read SAS**

For our example, you are going to use the `admission` dataset.


```r
PATH_sas <- 'https://github.com/thomaspernet/data_csv_r/blob/master/data/data_sas.sas7bdat?raw=true'

df <- read_sas(PATH_sas)
head(df)
```

```
## # A tibble: 6 x 4
##   ADMIT   GRE   GPA  RANK
##   <dbl> <dbl> <dbl> <dbl>
## 1    0.  380.  3.61    3.
## 2    1.  660.  3.67    3.
## 3    1.  800.  4.00    1.
## 4    1.  640.  3.19    4.
## 5    0.  520.  2.93    4.
## 6    1.  760.  3.00    2.
```
 
**Read STATA**

Next up are STATA data files; you can use `read_dta() for these. You use exactly the same dataset but store in `.dta` file.


```r
PATH_stata <- 'https://github.com/thomaspernet/data_csv_r/blob/master/data/stata.dta?raw=true'

df <- read_dta(PATH_stata)
head(df)
```

```
## # A tibble: 6 x 4
##   admit   gre   gpa  rank
##   <dbl> <dbl> <dbl> <dbl>
## 1    0.  380.  3.61    3.
## 2    1.  660.  3.67    3.
## 3    1.  800.  4.00    1.
## 4    1.  640.  3.19    4.
## 5    0.  520.  2.93    4.
## 6    1.  760.  3.00    2.
```

**Read SPSS**

You use the `read_sav()`function to open a SPSS  file. There is no difficulties in the task.


```r
PATH_spss <- 'https://github.com/thomaspernet/data_csv_r/blob/master/data/spss.sav?raw=true'

df <- read_sav(PATH_spss)
head(df)
```

```
## # A tibble: 6 x 4
##   admit   gre   gpa  rank
##   <dbl> <dbl> <dbl> <dbl>
## 1    0.  380.  3.61    3.
## 2    1.  660.  3.67    3.
## 3    1.  800.  4.00    1.
## 4    1.  640.  3.19    4.
## 5    0.  520.  2.93    4.
## 6    1.  760.  3.00    2.
```

### Best practices Data Import

When you want to import data into R, it is useful to implement following checklist. It will make it easy to import data correctly into R:

-	The typical format for a spreadsheet is to use the first rows as the header (usually variables name).
-	Avoid to name a dataset with blank spaces; it can lead to interpreting as a separate variable. Alternatively, prefer to use `_` or `-.
-	Short names are preferred
-	Do not include symbol in the name: i.e: `exchange_rate_$_eur`  is not correct. Prefer to name it:  `exchange_rate_dollar_eur`
-	Use `NA` for missing values otherwise; you need to clean the format later.

### Summary

Following table summarizes the function to use in order to import different types of file in R. The column one states the library related to the function. The last column refers to the default argument.

| Library | Objective       | Function     | Default Arguments                    |
|---------|-----------------|--------------|--------------------------------------|
| utils   | Read CSV file   | read.csv()   | file, header =,TRUE, sep = ","       |
| readxl  | Read EXCEL file | read_excel() | path, range = NULL, col_names = TRUE |
| haven   | Read SAS file   | read_sas()   | path                                 |
| haven   | Read STATA file | read_stata() | path                                 |
| haven   | Read SPSS fille | read_sav()   | path                                 |

Following table shows the different ways to import a selection with `read_excel() function.

| Function     | Objectives                            | Arguments                |
|--------------|---------------------------------------|--------------------------|
| read_excel() | Read n number of rows                 | n_max = 10               |
|              | Select rows and columns like in excel | range = "A1:D10"         |
|              | Select rows with indexes              | range= cell_rows(1:3)    |
|              | Select columns with letters           | range = cell_cols("A:C") |


