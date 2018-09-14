---
title: Data Visualization
author: Thomas
date: []
slug: data-visualization
categories: []
tags:
  - analysis
header:
  caption: ''
  image: ''
---

## Data Visualization

Graphs are the third part of the process of data analysis. The first part is about **data extraction**, the second part deals with **cleaning and manipulating the data**. At last, the data scientist may need to **communicate his results graphically**.

The job of the data scientist can be reviewed in the following picture 

-	The first task of a data scientist is to define a research question. This research question depends on the objectives and goals of the project.
-	After that, one of the most prominent tasks is the feature engineering. The data scientist needs to collect, manipulate and clean the data
-	When this step is completed, he can start to explore the dataset. Sometimes, it is necessary to refine and change the original hypothesis due to a new discovery. 

![](/project/data-visualization_files/17.png)

When the **explanatory** analysis is achieved, the data scientist has to consider the capacity of the reader to **understand the underlying concepts and models**. His results should be presented in a format that all stakeholders can understand.  One of the best methods to **communicate** the results is through a **graph**. Graphs are incredible tools to simplify complex analysis. 

This part of the tutorial focuses on how to make graphs/charts with R.

In this tutorial, you are going to use `ggplot2` package. This package is built upon the consistent underlying of the book *Grammar of graphics* written by Wilkinson, 2005. `ggplot2` is very flexible, incorporates many themes and plot specification at a high level of abstraction. With `ggplot2`, you can't plot 3-dimensional graphics and create interactive graphics. 

In `ggplot2`, a graph is composed of the following arguments:

- data
- aesthetic mapping
- geometric object
- statistical transformations
- scales
- coordinate system
- position adjustments
- faceting

You will learn how to control those arguments in the tutorial.

The basic syntax of `ggplot2` is:

```
ggplot(data, mapping = aes()) +
geometric object

arguments:
data:  Dataset used to plot the graph
mapping: Control the x and y-axis
geometric object: The type of plot you want to show. The most common objects are:

- Point: `geom_point()`
- Bar: `geom_bar()`
- Line: `geom_line()`
- Histogram: `geom_histogram()`
```

### Scatterplot

Let's see how ggplot works with the `mtcars` dataset. You start by plotting a scatterplot of the `mpg` variable and `drat` variable. 

**Basic scatter plot**


```r
library(ggplot2)
ggplot(mtcars, aes(x= drat, y = mpg))+
  geom_point()
```

![](/project/data-visualization_files/18_01.png)

Code Explanation

- You first pass the dataset mtcars to ggplot.  
- Inside the `aes()` argument, you add the x-axis and y-axis. 
- The `+` sign means you want R to keep reading the code. It makes the code more readable by breaking it.  
- Use `geom_point() for the geometric object.

**Scatter plot with groups**

Sometimes, it can be interesting to distinguish the values by a group of data (i.e. factor level data). 


```r
ggplot(mtcars, aes(x= mpg, y = drat))+
  geom_point(aes(color = factor(gear)))
```
![](/project/data-visualization_files/18_02.png)


Code Explanation

- The `aes()` inside the `geom_point()` controls the color of the group. The group should be a factor variable. Thus, you convert the variable `gear in a factor.
- Altogether, you have the code  `aes(color = factor(gear))` that changes the color of the dots.

**Change axis**

Rescale the data is a big part of the data scientist job. In rare occasion data comes in a nice bell shape. One solution to make your data less sensitive to outliers is to rescale them. 


```r
ggplot(mtcars, aes(x= log(mpg), y = log(drat))) +
  geom_point(aes(color = factor(gear)))
```

![](/project/data-visualization_files/18_03.png)


Code Explanation

- You transform the x and y variables in `log()` directly  inside the `aes()` mapping.

Note that any other transformation can be applied such as standardization or normalization.

**Scatter plot with fitted values**

You can add another level of information to the graph. You can plot the fitted value of a linear regression.


```r
library(ggplot2)
my_graph <- ggplot(mtcars, aes(x= log(mpg), y = log(drat)))+
  geom_point(aes(color = factor(gear)))+
  stat_smooth(method = "lm",
              col = "#C42126",  
              se = FALSE, 
              size = 1)
```


\begin{center}\includegraphics[width=0.65\linewidth]{/Users/Thomas/Dropbox/Learning/book_R/images/19} \end{center}

Code Explanation

- `graph`: You store your graph into the variable `graph`. It is helpful for further use or avoid too complex line of codes
- The argument `stat_smooth()` controls for the smoothing method
- `method = "lm"`: Linear regression
- `col = "#C42126"`: Code for the red color of the line
- `se = FALSE`:  Don't display the standard error
- `size = 1`: the size of the line is 1

Note that other smoothing methods are available

- `glm`
- `gam`
- `loess`: default value
- `rim`

**Add information to the graph**

So far, you haven't added information in the graphs. Graphs need to be informative. The reader should see the story behind the data analysis just by looking at the graph without referring additional documentation. Hence, graphs need good labels. You can add labels with `labs()`function.

The basic syntax for `lab()` is :

```
lab(title = "Hello World!")
argument:

- title: Control the title

It is possible to change or add title with:
  - subtitle: Add subtitle below title
  - caption: Add caption below the graph
  - x: rename x-axis
  - y: rename y-axis
  
Example:
lab(title = "Hello World!", subtitle = "My first plot")
````

**Add a title**

One mandatory information to add is obviously a title. 


```r
my_graph +
    labs(
        title = "Plot Mile per hours and drat, in log"
         )
```

![](/project/data-visualization_files/20.png)


Code Explanation

- `my_graph`: You use the graph you stored. It avoids rewriting all the codes each time you add new information to the graph.
- You wrap the title inside the `lab()`. 

**Add a title with a dynamic name**

A dynamic title is helpful to add more precise information in the title.

You can use the `paste()` function to print static text and dynamic text. The basic syntax of `paste()` is:

```
paste("This is a text", A)
arguments

- " ": Text inside the quotation marks are the static text
- A: Display the variable stored in A
- Note you can add as much static text and variable as you want. You need to separate them with a comma
```

Example:


```r
A <- 2010
paste("The first year is", A)
```

```
## [1] "The first year is 2010"
```

```r
B <- 2018
paste("The first year is", A, "and the last year is", B)
```

```
## [1] "The first year is 2010 and the last year is 2018"
```

You can add a dynamic name to your graph, namely the average of `mpg`.


```r
mean_mpg <- mean(mtcars$mpg)
my_graph +
  labs(
        title = paste("Plot Mile per hours and drat, in log. Average mpg is", mean_mpg)
        )
```

![](/project/data-visualization_files/21.png)


Code Explanation

- You create the average of `mpg` with `mean(mtcars$mpg)` stored in `mean_mpg` variable
- You use the `paste()` with `mean_mpg` to create a dynamic title returning the mean value of `mpg` 

**Add a subtitle**

Two additional details can make your graph more explicit. You are talking about the subtitle and the caption. The subtitle goes right below the title. The caption can inform about who did the computation and the source of the data. 


```r
my_graph +
  labs(
    title = 
      "Relation between Mile per hours and drat",
    subtitle = 
      "Relationship break down by gear class",
    caption = "Authors own computation"
  )
```


\begin{center}\includegraphics[width=0.65\linewidth]{/Users/Thomas/Dropbox/Learning/book_R/images/22} \end{center}

Code Explanation

- Inside the `lab()`, you added:
    - `title = "Relation between Mile per hours and drat"`: Add title
    - `subtitle = "Relationship break down by gear class"`: Add subtitle
    - `caption = "Authors own computation`: Add caption
    - You separate  each new information with a comma, `,`
- Note that you break the lines of code. It is not compulsory, and it only helps to read the code more easily

**Rename x-axis and y-axis**

Variables itself in the dataset might not always be explicit or by convention use the `_` when there are multiple words (i.e. `GDP_CAP`). You don't want such name to appear in your graph. It is important to change the name or add more details, like the units. 


```r
my_graph +
  labs(
    x = "Drat definition", 
    y = "Mile per hours",
    color= "Gear",
    title =
      "Relation between Mile per hours and drat",
    subtitle =
      "Relationship break down by gear class",
    caption = "Authors own computation"
  )
```

![](/project/data-visualization_files/23.png)


Code Explanation

- Inside the `lab()`, you added:
    - `x = "Drat definition"`: Change the name of x-axis
    - `y = "Mile per hours": Change the name of y-axis

**Control the scales**

You can control the scale of the axis. 

The function `seq()` is convenient when you need to create a sequence of number. The basic syntax is: 

```
seq(begin, last, by = x)
arguments:

- begin: First number of the sequence
- last: Last number of the sequence
- by= x: The step. For instance, if x is 2, the code adds 2 to `begin-1` until it reaches `last`
```

For instance, if you want to create a range from 0 to 12 with a step of 3, you will have four numbers, 0 4 8 12


```r
seq(0, 12,4)
```

```
## [1]  0  4  8 12
```

You can control the scale of the x-axis and y-axis as below


```r
my_graph +
  scale_x_continuous(breaks = seq(1, 3.6, by = 0.2))+
  scale_y_continuous(breaks = seq(1, 1.6, by = 0.1))+
  labs(
    x = "Drat definition", 
    y = "Mile per hours",
    color= "Gear",
    title = "Relation between Mile per hours and drat",
    subtitle = "Relationship break down by gear class",
    caption = "Authors own computation"
  )
```
![](/project/data-visualization_files/24.png)


Code Explanation

- The function `scale_y_continuous()` controls the **y-axis**  
- The function `scale_x_continuous()` controls the **x-axis**. 
- The parameter `breaks` controls the split of the axis. You can manually add the sequence of number or use the `seq()`function:
    - `seq(1, 3.6, by = 0.2)`: Create six numbers from 2.4 to 3.4 with a step of 3
    - `seq(1, 1.6, by = 0.1)`: Create seven numbers from 1 to 1.6 with a step of 1
    
**Theme**

Finally, R allows us to customize out plot with different themes. The library `ggplot2` includes eights themes:

- `theme_bw()`
- `theme_light()`
- `theme_classis()`
- `theme_linedraw()`
- `theme_dark()`
- `theme_minimal()`
- `theme_gray()`
- `theme_void()`


```r
my_graph+
  theme_classic()+
  labs(
    x = "Drat definition, in log", 
    y = "Mile per hours, in log",
    color= "Gear",
    title = "Relation between Mile per hours and drat",
    subtitle = "Relationship break down by gear class",
    caption = "Authors own computation"
  )
```

![](/project/data-visualization_files/1.png)


### Save Plots

After all these steps, it is time to save and share your graph. You add `ggsave('NAME OF THE FILE)` right after you plot the graph and it will be stored on the hard drive.

The graph is saved in the working directory. To check the working directory, you can run this code:


```r
directory <- getwd()
directory
```

```
## [1] "/Users/Thomas/Dropbox/Learning/book_R"
```

Let's plot your fantastic graph, saves it and checks the location


```r
my_graph +
  theme_classic()+
  labs(
    x = "Drat definition, in log", 
    y = "Mile per hours, in log",
    color= "Gear",
    title = "Relation between Mile per hours and drat",
    subtitle = "Relationship break down by gear class",
    caption = "Authors own computation"
  )
```

![](/project/data-visualization_files/2.png)

```r
ggsave("my_fantastic_plot.png")
```

```
## Saving 6.5 x 4.5 in image
```


*note*: For pedagogical  purpose only, you created a function called `open_folder()` to open the directory folder for you. You just need to run the code below and see where the picture is stored. You should see a file names `my_fantastic_plot.png`.


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

### Summary

You can summarize the arguments to create a scatter plot in the table below:

| Objective                     | Code                                                                                                                                               |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Basic scatter plot            | ggplot(df, aes(x = x1, y = y)) +  geom_point()                                                                                                     |
| Scatter plot with color group | ggplot(df, aes(x = x1, y = y)) +  geom_point(aes(color = factor(x1)) + stat_smooth(method = "lm")                                                  |
| Add fitted values             | ggplot(df, aes(x = x1, y = y)) +  geom_point(aes(color = factor(x1))                                                                               |
| Add title                     | ggplot(df, aes(x = x1, y = y)) +  geom_point() + labs(title = paste("Hello Guru99"))                                                               |
| Add subtitle                  | ggplot(df, aes(x = x1, y = y)) +  geom_point() + labs(subtitle = paste("Hello Guru99"))                                                            |
| Rename x                      | ggplot(df, aes(x = x1, y = y)) +  geom_point() + labs(x = "X1")                                                                                    |
| Rename y                      | ggplot(df, aes(x = x1, y = y)) +  geom_point() + labs(y = "y1")                                                                                    |
| Control the scale             | ggplot(df, aes(x = x1, y = y)) +  geom_point() + scale_y_continuous(breaks = seq(10, 35, by = 10)) + scale_x_continuous(breaks = seq(2, 5, by = 1) |
| Create logs                   | ggplot(df, aes(x =log(x1), y = log(y))) +  geom_point()                                                                                            |
| Theme                         | ggplot(df, aes(x = x1, y = y)) +  geom_point() + theme_classic()                                                                                   |
| Save                          | ggsave("my_fantastic_plot.png")                                                                                                                    |

Complete code:

```
ggplot(mtcars, aes(x= log10(x1), y = log10(x2)))+
  geom_point(aes(color = factor(x3)))+
  geom_smooth(se = FALSE)+
  theme_classic()+
  labs(
    x = "X1, in log", 
    y = "y, in log",
    color= "Gear",
    title = "Hello world",
    subtitle = "My fantastic plot",
    caption = "Made by Hello World"
  )
ggsave("my_fantastic_plot.png")
```

### Bar chart

A bar chart is a great way to display categorical variables in the x-axis. This type of graph denotes two aspects in the y-axis.

1.	The first one counts the number of occurrence between groups. 
2.	The second one shows a summary statistic (min, max, average, and so on) of a variable in the y-axis. 


You will use the `mtcars` dataset with has the following variables:

- `cyl`: Number of the cylinder in the car. Numeric variable
- `am`: Type of transmission. 0 for automatic and 1 for manual. Numeric variable
- `mpg: Miles per gallon. Numeric variable

To create graph in R, you can use the library `ggplot` which creates ready-for-publication graphs. The basic syntax of this library is:

```
ggplot(data, mapping = aes()) +
geometric object

arguments:
data:  dataset used to plot the graph
mapping: Control the x and y axis
geometric object: The type of plot you want to show. The most common object are:

- Point: `geom_point()`
- Bar: `geom_bar()`
- Line: `geom_line()`
- Histogram: `geom_histogram()`
```

In this tutorial, you are interested with the geometric object `geom_bar()` that create the **bar chart**.

**Bar chart: count**

Our first graph shows the frequency of cylinder with `geom_bar()`. The code below is the most basic syntax. 


```r
library(ggplot2)
# Most basic bar chart
ggplot(mtcars, aes(x=factor(cyl)))+
  geom_bar()
```

![](/project/data-visualization_files/3.png)


Code Explanation

-	You pass the dataset `mtcars` to ggplot. 
-	Inside the `aes()` argument, you add the x-axis as a factor variable.
-	The `+` sign means you want R to keep reading the code. It is makes the code more readable by breaking it.
-	Use `geom_bar()` for the geometric object.

*note*: make sure you convert the variables into a factor otherwise R treats the variables as numeric. See the example below.

![](/project/data-visualization_files/26.png)


**Customize the graph**

Four arguments can be passed to customize the graph:

```
- `stat`: Control the type of formatting. By default, `bin` to plot a count in the y-axis. For continuous value, pass `stat = "identity"`
- `alpha`: Control density of the color
- `fill`: Change the color of the bar
- `size`: Control the size the bar
```

**Change the color of the bars**

You can change the color of the bars. Note that the colors of the bars are all similar.


```r
# Change the color of the bars
ggplot(mtcars, aes(x=factor(cyl)))+ 
  geom_bar(fill = "coral") +
  theme_classic()
```

![](/project/data-visualization_files/4.png)

Code Exlanation

- The colors of the bars are controlled by the `aes()` mapping inside the geometric object (i.e. not in the `ggplot()`). You can change the color with the `fill` argument. Here, you choose the `coral` color.

you can use this code:


```r
grDevices::colors()
```

to see all the colors available in R. There are around 650 colors.

**Change the intensity**

You can increase or decrease the intensity of the bars' color


```r
# Change intensity
ggplot(mtcars, aes(factor(cyl)))+ 
  geom_bar(fill = "coral",
           alpha = 0.5) +
  theme_classic()
```

![](/project/data-visualization_files/5.png)

Code Explanation

- To increase/decrease the intensity of the bar, you can change the value of the `alpha`. A large `alpha` increases the intensity and low `alpha` reduces the intensity. `alpha` ranges from 0 to 1. If 1, then the color is the same as the palette. If 0, color is white. You choose `alpha = 0.1`.

**Color by groups**

You can change the colors of the bars, meaning one different color for each group. For instance, `cyl` variable has three levels, then you can plot the bar chart with three colors.


```r
# Color by group
ggplot(mtcars, aes(factor(cyl), fill = factor(cyl)))+
  geom_bar()
```

![](/project/data-visualization_files/6.png)


Code Explanation

- The argument `fill` inside the `aes()` allows changing the color of the bar. You change the color by setting `fill = x-axis variable`. In your example, the x-axis variable is `cyl`; `fill = factor(cyl)`

**Add a group in the bars**

You can further split the y-axis based on another factor level. For instance, you can count the number of automatic and manual transmission based on the cylinder type. 

You will proceed as follow:

- Step 1: Create the data frame with `mtcars` dataset
- Step 2: Label the `am` variable with `auto` for automatic transmission and `man` for manual transmission. Convert `am` and `cyl` as a factor so that you don't need to use `factor()` in the `ggplot()` function. 
- Step 3: Plot the bar chart to count the number of transmission by cylinder


```r
library(dplyr)
# Step 1
data <- mtcars %>%
# Step 2  
        mutate(am = factor(am, labels= c("auto", "man")),
                           cyl = factor(cyl))
```

You have the dataset ready, you can plot the graph;


```r
# Step 3
ggplot(data, aes(x= cyl, fill = am))+
  geom_bar()+
  theme_classic()
```
![](/project/data-visualization_files/7.png)

Code Explanation

- The `ggpplot()` contains the dataset `data` and the `aes()`.
- In the `aes()` you include the variable x-axis and which variable is required to fill the bar (i.e. `am`)
- `geom_bar()`: Create the bar chart

The mapping will fill the bar with two colors, one for each level. It is effortless to change the group by choosing other factor variables in the dataset.

**Bar chart in percentage**

You can visualize the bar in percentage instead of the raw count.


```r
# Bar chart in percentage
ggplot(data, aes(x= cyl,fill = am))+
  geom_bar(position = "fill")+
  theme_classic()
```

![](/project/data-visualization_files/8.png)


Code Explanation
 
- Use `position = "fill"` in the `geom_bar()` argument to create a graphic with percentage in the y-axis.

**Side by side bars**

It is easy to plot the bar chart with the group variable side by side..


```r
# Bar chart side by side
ggplot(data, aes(x= cyl, fill = am))+
  geom_bar(position=position_dodge())+
  theme_classic()
```

![](/project/data-visualization_files/9.png)

Code Explanation

- `position=position_dodge()`: Explicitly tells how to arrange the bars

### Histogram

In the second part of the bar chart tutorial, you can represent the group of variables with values in the y-axis.

Your objective is to create a graph with the average mile per gallon for each type of cylinder. To draw an informative graph, you will follow these steps:

- Step 1: Create a new variable with the average mile per gallon by cylinder
- Step 2: Create a basic histogram
- Step 3: Change the orientation 
- Step 4: Change the color
- Step 5: Change the size
- Step 6: Add labels to the graph

**Step 1: Create a new variable**

You create a data frame named `data_histogram` which simply returns the average miles per gallon by the number of cylinders in the car. 
You call this new variable `mean_mpg`, and you round the mean with two decimals. 


```r
# Step 1
data_histogram <- mtcars %>%
        mutate(cyl = factor(cyl)) %>%
        group_by(cyl)%>%
        summarise(mean_mpg = round(mean(mpg), 2))
```

**Step 2: Create a basic histogram**

You can plot the histogram. It is not ready to communicate to be delivered to client but gives us an intuition about the trend.


```r
ggplot(data_histogram, aes(x= cyl, y = mean_mpg))+
  geom_bar(stat="identity")
```

![](/project/data-visualization_files/10.png)


Code Explanation

- The `aes()` has now two variables. The `cyl` variable refers to the x-axis, and the `mean_mpg` is the y-axis.
- You need to pass the argument `stat="identity"` to refer the variable in the y-axis as a numerical value. `geom_bar` uses `stat="bin"` as default value. 

**Step 3: Change the orientation**

You change the orientation of the graph from vertical to horizontal.


```r
ggplot(data_histogram, aes(x= cyl, y = mean_mpg))+
  geom_bar(stat="identity") +
  coord_flip()
```



![](/project/data-visualization_files/11.png)

Code Explanation

- You can control the orientation of the graph with `coord_flip()`.

**Step 4: Change the color**

You can differentiate the colors of the bars according to the factor level of the x-axis variable.


```r
ggplot(data_histogram, aes(x= cyl, y = mean_mpg, fill = cyl))+
         geom_bar(stat="identity")+
         coord_flip()+
  theme_classic()
```

![](/project/data-visualization_files/12.png)

Code Explanation

- You can plot the graph by groups with the `fill= cyl` mapping. R takes care automatically of the colors based on the levels of `cyl` variable.

**Step 5: Change the size**

To make the graph looks prettier, you reduce the width of the bar.


```r
graph <- ggplot(data_histogram, aes(x= cyl, y = mean_mpg, fill = cyl)) +
  geom_bar(stat="identity",
           width=0.5) +
  coord_flip()+
  theme_classic()

graph
```

![](/project/data-visualization_files/13.png)

Code Explanation

-  The `width` argument inside the `geom_bar()` controls the size of the bar. Larger value increases the width.
- Note, you store the graph in the variable `graph`. you do so because the next step will not change the code of the variable `graph`. It improves the readability of the code.

**Step 6: Add labels to the graph**

The last step consists to add the value of the variable `mean_mpg` in the label. 


```r
graph +
  geom_text(aes(label=mean_mpg), 
            hjust=1.5, 
            color="white", 
            size=3)+
  theme_classic()
```
![](/project/data-visualization_files/14.png)

Code Explanation

- The function `geom_text()` is useful to control the aesthetic of the text. 
    - `label=`: Add a label inside the bars 
    - `mean_mpg`: Use the variable `mean_mpg` for the label 
- `hjust` controls the location of the label. Values closed to 1 displays the label at the top of the bar, and higher values bring the label to the bottom. If the orientation of the graph is vertical, change `hjust` to  `vjust`.
- `color="white"`: Change the color of the text. Here you use the white color.
- `size=3`: Set the size of the text.

### Summary

A bar chart is useful when the x-axis is a categorical variable. The y-axis can be either a count or a summary statistic. The table below summarizes how to control bar chart with `ggplot2`:

| Objective                          | code                                                                                    |
|------------------------------------|-----------------------------------------------------------------------------------------|
| Count                              | ggplot(df, eas(x= factor(x1)) + geom_bar()                                              |
| Count with different color of fill | ggplot(df, eas(x= factor(x1), fill = factor(x1))) + geom_bar()                          |
| Count with groups, stacked         | ggplot(df, eas(x= factor(x1), fill = factor(x2))) + geom_bar(position=position_dodge()) |
| Count with groups, side by side    | ggplot(df, eas(x= factor(x1), fill = factor(x2))) + geom_bar()                          |
| Count with groups, stacked in %    | ggplot(df, eas(x= factor(x1), fill = factor(x2))) + geom_bar(position=position_dodge()) |
| Values                             | ggplot(df, eas(x= factor(x1)+ y = x2) + geom_bar(stat="identity")                       |
### Box plot

You can use the geometric object `geom_boxplot()` from `ggplot2` library to draw a box plot. Box plot helps to **visualise the distribution of the data by quartile and detect the presence of outliers.**

You will use the `airquality` dataset to introduce box plot with `ggplot`. This dataset measures the airquality of New York from May to September 1973.
The dataset contains 154 observations. You will use the following variables:

-	`Ozone`: Numerical variable
-	`Wind`:	Numerical variable
-	`Month`: May to September. Numerical variable

Before you start to create your first box plot, you need to manipulate the data as follow:

- Step 1: Import the data
- Step 2: Drop unnecessary variables
- Step 3: Convert `Month` in factor level
- Step 4: Create a new categorical variable dividing the month with three level: `begin`, `middle` and `end`.
- Step 5: Remove missing observations

All these steps are done with `dplyr` and the pipeline operator `%>%`.


```r
library(dplyr)
library(ggplot2)
# Step 1
data_air <- airquality %>%
# Step 2  
            select(-c(Solar.R, Temp)) %>%
# Step 3  
            mutate(Month = factor(Month, order = TRUE, labels = c("May", "June", "July", "August", "September")),
#Step 4                   
                   day_cat = factor(ifelse(Day < 10, "Begin",
                                    ifelse(Day <20, "Middle", "End"))))
```

A good practice is to check the structure of the data with the function  `glimpse()`.


```r
glimpse(data_air)
```

```
## Observations: 153
## Variables: 5
## $ Ozone   <int> 41, 36, 12, 18, NA, 28, 23, 19, 8, NA, 7, 16, 11, 14, ...
## $ Wind    <dbl> 7.4, 8.0, 12.6, 11.5, 14.3, 14.9, 8.6, 13.8, 20.1, 8.6...
## $ Month   <ord> May, May, May, May, May, May, May, May, May, May, May,...
## $ Day     <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,...
## $ day_cat <fct> Begin, Begin, Begin, Begin, Begin, Begin, Begin, Begin...
```

There are `NA`'s in the dataset. Removing them is wise.


```r
# Step 5
data_air_nona <- data_air %>% na.omit()
```

### Basic box plot
Let's plot the basic box plot with the distribution of ozone by month. 


```r
# Store the graph 
box_plot <- ggplot(data_air_nona, aes(x=Month, y=Ozone))
# Add the geometric object box plot
box_plot +
  geom_boxplot()
```

![](/project/data-visualization_files/15.png)
Code Explanation

- First step: Store the graph for further use
    - `box_plot`: You store your graph into the variable `box_plot` It is helpful for further use or avoid too complex line of codes
- Second step: Add your geometric object box plot
    -	You pass the dataset `data_air_nona` to ggplot. 
    -	Inside the `aes()` argument, you add the x-axis and y-axis.
    -	The `+` sign means you want R to keep reading the code. It makes the code more readable by breaking it.
    -	Use `geom_boxplot()` to create a box plot.

**Change side of the graph**

You can flip the side of the graph.


```r
box_plot +
  geom_boxplot()+
  coord_flip()
```
![](/project/data-visualization_files/27.png)

Code Explanation

-	`box_plot`: You use the graph you stored. It avoids rewriting all the codes each time you add new information to the graph
- `geom_boxplot()`: Create the box plot
- coord_flip()`: Flip the side of the graph 

**Change color of outlier**

You can change the color, shape and size of the outliers.


```r
box_plot +
  geom_boxplot(outlier.colour="red",
               outlier.shape=2,
               outlier.size=3) +
  theme_classic()
```

![](/project/data-visualization_files/28.png)

Code Explanation

- `outlier.colour="red"`: Control the color of the outliers
- `outlier.shape=2`: Change the shape of the outlier. 2 refers to triangle
- `outlier.size=3`: Change the size of the triangle. The size is proportional to the number.

**Add a summary statistic**

You can add a summary statistic to the box plot. 


```r
box_plot+
  geom_boxplot() +
   stat_summary(fun.y=mean,
                geom = "point",
                size=3,
                color ="steelblue") +
  theme_classic()
```


![](/project/data-visualization_files/29.png)

Code Explanation

- `stat_summary()` allows adding a summary to the box plot
- The argument `fun.y` controls the statistics returned. You will use `mean`.  
- Note: Other statistics are available such as `min` and `max`.More than one statistics can be exhibited in the same graph.
- `geom = "point"`: Plot the average with a point
- `size=3`: Size of the point
- `color ="steelblue"`: Color of the points

### box plot with dots

In the next plot, you add the dot plot layers. Each dot represents an observation.


```r
box_plot+
  geom_boxplot()  +
  geom_dotplot(binaxis='y',
               dotsize=1,
               stackdir='center')+
  theme_classic()
```


![](/project/data-visualization_files/30.png)

Code Explanation

- `geom_dotplot()` allows adding dot to the bin width
- `binaxis='y'`: Change the position of the dots along the y-axis. By default, x-axis
- `dotsize=1`: Size of the dots
- `stackdir='center'`: Way to stack the dots: Four values: 
    - "up" (default), 
    - "down"
    - "center" 
    - "centerwhole"

### Control aesthetic of the box plot

**Change the color of the box**

You can change the colors of the group. 


```r
ggplot(data_air_nona, aes(x = Month, y = Ozone, color= Month)) +
  geom_boxplot()  +
  theme_classic()
```



![](/project/data-visualization_files/31.png)

Code Explanation

-  The colors of the groups are controlled in the `aes()` mapping. You can use `color= Month` to change the color of the box according to the months

**Box plot with multiple groups**

It is also possible to add multiple groups. You can visualize the difference in the air quality according to the day of the measure. 


```r
ggplot(data_air_nona, aes(Month, Ozone)) +
  geom_boxplot(aes(fill= day_cat))+
  theme_classic()
```

![](/project/data-visualization_files/32.png)

Code Explanation

- The `aes()` mapping of the geometric object controls the groups to display (this variable has to be a factor). 
- `aes(fill= day_cat)` allows creating three boxes for each month in the x-axis

**box plot with jittered dots**

Another way to show the dot is with jittered points. This is a convenient way to visualize points with a categorical variable. 

This method avoids the overlapping of the discrete data. 


```r
box_plot +
  geom_boxplot()  +
  geom_jitter(shape=15, 
              color = "steelblue",
              position=position_jitter(width = 0.21))+
  theme_classic()
```

![](/project/data-visualization_files/33.png)

Code Explanation

- `geom_jitter()` adds a little decay to each point. 
- `shape=15` changes the shape of the points. 15 represents the squares
- `color = "steelblue"`: Change the color of the point
- `position=position_jitter(width = 0.21)`: Way to place the overlapping points. `position_jitter(width = 0.21)` means you move the points by 20 percent from the x-axis. By default, 40 percent.  

You can see the difference between the first graph with the jitter method and the second with the point method.


```r
box_plot +
  geom_boxplot()  +
  geom_point(shape=5, 
              color = "steelblue")+
  theme_classic()
```

![](/project/data-visualization_files/34.png)

**Notched box plot**

You can use an interesting feature of `geom_boxplot()`, a notched box plot. The notch plot narrows the box around the median. The main purpose of a notched box plot is to compare the significance of the median between groups. There are strong evidence two groups have different medians when the notches do not overlap. A notch is computed as follow:

$$
median  \pm 1.57 * \frac{IQR}{\sqrt{n}}
$$

with $IQR$  is the interquartile and $n$ number of observations. 


```r
box_plot +
  geom_boxplot(notch=TRUE) +
  theme_classic()
```


![](/project/data-visualization_files/35.png)

Code Explanation

- `geom_boxplot(notch=TRUE)`: Create a notched box plot

### Summary

You can summarize the different types of box plot in the table below:

| Objective                   | Code                                                                                             |
|-----------------------------|--------------------------------------------------------------------------------------------------|
| Basic box plot              | ggplot(df, aes( x = x1, y =y)) + geom_boxplot()                                                  |
| flip the side               | ggplot(df, aes( x = x1, y =y)) + geom_boxplot() + coord_flip()                                   |
| Notched box plot            | ggplot(df, aes( x = x1, y =y)) + geom_boxplot(notch=TRUE)                                        |
| Box plot with jittered dots | ggplot(df, aes( x = x1, y =y)) + geom_boxplot() +  geom_jitter(position = position_jitter(0.21)) |