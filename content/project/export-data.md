---
title: Export Data
author: Thomas
date: []
slug: export-data
categories: []
tags:
  - preparation
header:
  caption: ''
  image: ''
---

## Export data

In this chapter, we will learn how to export data from R environment to different formats. 

To export data to the hard drive, you need a path and an extension. First of all, the path is the location of the data will be stored. In this tutorial, you will see how to store data on:

  - The hard drive
- Google Drive
- Dropbox

Secondly, R allows the users to export the data into different types of files. We cover the essential file's extension:

- csv
- xlsx
- RDS
- SAS
- SPSS 
- STATA

Overall, it is not difficult to export data from R.

### Hard drive

To begin with, you can save the data directly into the working directory. The following code prints the path of your working directory:


```r
directory <- getwd()
directory
```

```
## [1] "/Users/Thomas/Dropbox/Learning/book_R"
```

By default, file will be saved in this path. You can, of course, set a different path. For instance, you can change the path to the download folder. We will see in the function description how to change the path.


For Mac OS:

```
/Users/USERNAME/Downloads/
```

For Windows:

```
C:\Users\USERNAME\Downloads\
```

First of all, let's import the `mtcars` dataset and get the mean of `mpg` and `disp` grouped by `gear`. 


```r
library(dplyr)
df <- mtcars %>%
  select(mpg, disp, gear) %>%
  group_by(gear) %>%
  summarise(mean_mpg = mean(mpg), mean_disp = mean(disp))
df
```

```
## # A tibble: 3 x 3
##    gear mean_mpg mean_disp
##   <dbl>    <dbl>     <dbl>
## 1    3.     16.1      326.
## 2    4.     24.5      123.
## 3    5.     21.4      202.
```

The table contains three rows and three columns. You can create a CSV file with the function `write.csv()`.

### Export CSV

The basic syntax is: 

```
write.csv(df, path, sep = "\t")
arguments

- df: Dataset to save. Need to be the same name of the data frame in the environment.
- path: A string. Set the destination path. Path+ filename + extention i.e. "/Users/USERNAME/Downloads/mydata.csv" or the filename + extension if the folder is the same as the working directory
- Note: A decimal is used to the separator. 
```

Example:


  ```r
  write.csv(df, "table_car.csv")
  ```

Code Explanation

- `write.csv(df, "table_car.csv")`: Create a CSV file in the hard drive:
- `df`:  name of the data frame in the environment
- `"table_car.csv"`: Name the file table_car and store it as csv

*note*: You can use the function `write.csv2()` to separate the rows with a semicolon.


```r
write.csv2(df, "table_car.csv")
```

*note*: For pedagogical  purpose only, we created a function called `open_folder()` to open the directory folder for you. You just need to run the code below and see where the csv file is stored. You should see a file names `table_car.csv`.


```r
# Run this code to create the function
open_folder <- function(dir){
  if (.Platform['OS.type'] == "windows"){
    shell.exec(dir)
  } else {
    system(paste(Sys.getenv("R_BROWSER"), dir))
  }
}

# Call the function to open the folder
open_folder(directory)
```

### Export excel file

Export data to Excel is trivial for Windows users and trickier for Mac OS user. Both users will use the library `xlsx` to create an Excel file. The slight difference comes from the installation of the library. Indeed, the library `xlsx` uses Java to create the file, Java is not installed by default on the mac OS machine.

**Windows users**

If you are a Windows users, you can install the library directly with conda:

  ```
conda install -c r r-xlsx
  ```

Once the library installed, you can use the function `write.xlsx()`. A new Excel workbook is created in the working directory


```r
library(xlsx)
write.xlsx(df, "table_car.xlsx")
```

**Mac users**

If you are a Mac OS user, you need to follow these steps:

- Step 1: Install latest version of Java
- Step 2: Install library `rJava`
- Step 3: Install library `xlsx`

**Step 1**

The easiest way to install Java might be with **Homebrew**. If you have Homebrew already installed on your machine, you can copy and paste the following code to the terminal:

  ```
brew cask install java
  ```

<img src="/project/export-data/30.png" width="55%" style="display: block; margin: auto;" />

If you don't have Homebrew already on your machine, you need to install it. The next two lines of code install Homebrew and Java 

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew cask install java
```

You have now the latest version of Java on your machine. 

![](/projet/export-data_files/31.png)
<img src="/project/export-data/31.png" width="55%" style="display: block; margin: auto;" />

You can go back to Rstudio and check which version of Java is installed.


```r
system("java -version")
```

At the time of the tutorial, the latest version of Java is 9.0.4.

**Step 2**

You need to install `rjava` in R. We recommended you to install R and Rstudio with Anaconda. Anaconda managed the dependencies between libraries. In this sense, Anaconda will do all the job to install correctly and efficiently `rjava`. 

First of all, you need to update conda


```
conda - conda update
```

and then install `rjava`

- `rjava`: Add Java to R.
- [Anaconda](https://anaconda.org/r/r-rjava): `conda install -c r r-rjava`

You should be able to open `rjava` in Rstudio


```r
library(rJava)
```

**Step 3**

Finally, it is time to install `xlsx`.

- `xlsx`: Export file to excel.
- [Anaconda](https://anaconda.org/r/r-xlsx): `conda install -c r r-xlsx`

Just as the windows users, you can save data with the function `write.xlsx()`


```r
library(xlsx)
```

```
## Loading required package: xlsxjars
```

```r
write.xlsx(df, "table_car.xlsx")
```

### Export to different software

Exporting data to different software is as simple as importing them. The library `haven` provides a convenient way to export data to 

- spss
- sas
- stata

First of all, import the library. If you don't have `haven`, you can go [here](https://anaconda.org/conda-forge/r-haven) to install it.


```r
library(haven)
```

**SPSS file**

Below is the code to export the data to SPSS software:


  ```r
  write_sav(df, "table_car.sav")
  ```

**Export SAS file**

Just as simple as spss, you can export to sas


```r
write_sas(df, "table_car.sas7bdat")
```

**Export STATA file**

Finally, `haven` library allows to write `.dta` file.


```r
write_dta(df, "table_car.dta")
```

**R** 

If you want to save a data frame or any other R object, you can use the `save()` function.


```r
save(df, file = 'table_car.RData')
```

### Interact with the cloud services

Last but not least, R is equipped with amazing libraries to interact with the cloud computing. The last part of this tutorial deals with export/import files from:

  - Google Drive
- Dropbox

*note*: This part of the tutorial assumes you have an account with Google and Dropbox. If not, you can easily create one for

- Gmail: [](https://accounts.google.com/SignUp?hl=en)
- Dropbox: [](https://www.dropbox.com/h)

### Google Drive

You need to install the library `googledrive` to access the function allowing to interact with Google Drive.

The library is not yet available at Anaconda. You want to use Anaconda to manage all your libraries. To install libraries out of the conda libraries, you need to check the library path (i.e. where R goes to find libraries):


```r
lib <- .libPaths()
lib
```

```
## [1] "/Users/Thomas/anaconda3/envs/hello-r/lib/R/library"
```

You should see `anaconda3/lib/R/library`. That is where you want to install your library. Copy this path. 

For non-conda user, installing a library is easy, you can use the function `install.packages('NAME OF PACKAGE)` with the name of the package inside the parenthesis. Don't forget the `' '`. Note that, R is supposed to install the package in the `libPaths() automatically. It is worth to see it in action. 


```r
install.packages("googledrive", lib)
```

and you open the library.


```r
library(googledrive)
```

**Upload to Google Drive**

To upload a file to Google drive, you need to use the function `drive_upload()`.

Each time you restart Rstudio, you will be prompted to allow access `tidyverse` to Google Drive. 

The basic syntax of `drive_upload()` is 

```
drive_upload(file, path = NULL, name = NULL)
arguments:
  
  - file: Full name of the file to upload (i.e. including the extension)
- path: Location of the file
- name: You can rename it as you wish. By default, it is the local name. 
```

After you launch the code, you need to confirm several questions


```r
drive_upload("table_car.csv", name = "table_car")
```

```
## Local file:
##   * table_car.csv
## uploaded into Drive file:
##   * table_car: 1GIaR4DCy1iTbe6Wh-W1onh5CFK9ckkIT
## with MIME type:
##   * text/csv
```

You type 1 in the console to confirm the access


<img src="/project/export-data/32.png" width="55%" style="display: block; margin: auto;" />

Then, you are redirected to Google API to allow the access. Click Allow.


<img src="/project/export-data/33.png" width="55%" style="display: block; margin: auto;" />

When the authentication is completed, you can quit your browser.


<img src="/project/export-data/34.png" width="55%" style="display: block; margin: auto;" />

In the Rstudio's console, you can see the summary of the step done. Google successfully uploaded the file located locally on the Drive. Google assigned an ID to each file in the drive. 


<img src="/project/export-data/35.png" width="55%" style="display: block; margin: auto;" />

You can see this file in Google Spreadsheet.


```r
drive_browse("table_car")
```


<img src="/project/export-data/36.png" width="55%" style="display: block; margin: auto;" />

You will be redirected to Google Spreadsheet


<img src="/project/export-data/37.png" width="55%" style="display: block; margin: auto;" />

**Import from Google Drive**

Upload a file from Google Drive with the ID is convenient. If you know the file name, you can get its ID as follow:

*note*: Depending on your internet connection and the size of your Drive, it takes times.


```r
x <- drive_get("table_car")
```

```
## Warning: 'collapse' is deprecated.
## Use 'glue_collapse' instead.
## See help("Deprecated") and help("glue-deprecated").

## Warning: 'collapse' is deprecated.
## Use 'glue_collapse' instead.
## See help("Deprecated") and help("glue-deprecated").
```

```r
as_id(x)
```

```
## [1] "1GIaR4DCy1iTbe6Wh-W1onh5CFK9ckkIT"
## attr(,"class")
## [1] "drive_id"
```


<img src="/project/export-data/38.png" width="55%" style="display: block; margin: auto;" />

You stored the ID in the variable `x`. The function `drive_download()` allows downloading a file from Google Drive. 

The basic syntax is: 

```
drive_download(file, path = NULL, overwrite = FALSE)
arguments:

- file:  Name or id of the file to download
- path: Location to download the file. By default, it is downloaded to the working directory, and the name in Google Drive
- overwrite = FALSE: If the file already exists, don't overwrite it. If set to TRUE, the old file is erased and replaced by the new one.
```

You can finally download the file:


```r
download_google <- drive_download(as_id(x), overwrite = TRUE)
```

```
## File downloaded:
##   * table_car
## Saved locally as:
##   * table_car
```

Code Explanation

- `drive_download()`:  Function to download a file from Google Drive
- `as_id(x)`:  Use the ID to browse the file in Google Drive
- `overwrite = TRUE`: If file exists, overwrite it, else execution halted
To see the name of the file locally, you can use:
  
  
  ```r
  google_file <- download_google$local_path
  google_file
  ```
  
  ```
  ## [1] "table_car"
  ```

It is trivial to open the file. The file is stored in your working directory. Remember, you need to add the extenstion of the file to open it in R. You can create the full name with the function `paste()` (i.e. `table_car.csv`)


```r
path <- paste(google_file, ".csv", sep ="")
google_table_car <- read.csv(path)
google_table_car
```

```
##   X gear mean_mpg mean_disp
## 1 1    3 16.10667  326.3000
## 2 2    4 24.53333  123.0167
## 3 3    5 21.38000  202.4800
```

Finally, you can remove the file from your Google drive. 


```r
## clean-up
drive_find("table_car") %>% drive_rm()
```

### Export to Dropbox

R interacts with Dropbox via the `rdrop2` library. The library is not available at Anaconda as well. You can install it via the console


```r
install.packages('rdrop2')
```


```r
library(rdrop2)
```

You need to provide a temporary access to Dropbox with your credential. After the identification is done, R can create, remove upload and download to your Dropbox.

First of all, you need to give the access to your account. The credentials are cached during all session. 


```r
drop_auth()
```

You will be redirected to Dropbox to confirm the authentification. You need to sign in to link Dropbox with `rdrop2`


<img src="/project/export-data/39.png" width="55%" style="display: block; margin: auto;" />

You can create a folder with the function `drop_create()`. 

- `drop_create('my_first_drop')`: Create a folder in the first branch of Dropbox
- `drop_create('First_branch/my_first_drop')`: Create a folder inside the existing `First_branch` folder.


```r
drop_create('my_first_drop')
```

To upload the .csv file into your Dropbox, use the function `drop_upload()`. 

Basic syntax: 

```
drop_upload(file, path = NULL, mode = "overwrite")
arguments:
  
  - file: local path
- path: Path on Dropbox 
- mode = "overwrite":  By default, overwrite an existing file. If set to `add`, the upload is not completed.
```


```r
drop_upload('table_car.csv', path = "my_first_drop")
```

You can read the csv file from Dropbox with the function `drop_read_csv()`


```r
dropbox_table_car <- drop_read_csv("my_first_drop/table_car.csv") 
dropbox_table_car
```

```
##   X gear mean_mpg mean_disp
## 1 1    3 16.10667  326.3000
## 2 2    4 24.53333  123.0167
## 3 3    5 21.38000  202.4800
```

When you are done using the file and want to delete it. You need to write the path of the file in the function `drop_delete()`


```r
drop_delete('my_first_drop/table_car.csv')
```

It is also possible to delete a folder


```r
drop_delete('my_first_drop')
```

### Summary 

We can summarize all the functions in the table below

| Library     | Objective                     | Function           |
|-------------|-------------------------------|--------------------|
| base        | Export csv                    | write.csv()        |
| xlsx        | Export excel                  | write.xlsx()       |
| haven       | Export spss                   | write_sav()        |
| haven       | Export sas                    | write_sas()        |
| haven       | Export stata                  | write_dta()        |
| base        | Export R                      | save()             |
| googledrive | Upload Google Drive           | drive_upload()     |
| googledrive | Open in Google Drive          | drive_browse()     |
| googledrive | Retrieve file ID              | drive_get(as_id()) |
| googledrive | Dowload from Google Drive     | download_google()  |
| googledrive | Remove file from Google Drive | drive_rm()         |
| rdrop2      | Authentification              | drop_auth()        |
| rdrop2      | Create a folder               | drop_create()      |
| rdrop2      | Upload to Dropbox             | drop_upload()      |
| rdrop2      | Read csv from Dropbox         | drop_read_csv      |
| rdrop2      | Delete file from Dropbox      | drop_delete()      |



