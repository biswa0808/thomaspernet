---
title: Conditional Statement
author: Thomas
date: []
slug: conditional-statement
categories: []
tags:
  - program
header:
  caption: ''
  image: ''
---

## Conditional Statement

### One condition statement

An `if else` statement is a great tool for the developer trying to return an output based on condition. In R, the syntax is:

```
if (condition) {
   expr1 
   } else {
   expr2   
   }
```

<img src="/project/conditional-statement_files/3.png" width="55%" style="display: block; margin: auto;" />

You want to examine whether a variable stored as `quantity` is above 20. If `quantity`is greater than 20, the code print `You sold a lot!` otherwise `Not enough for today`. 


```r
# Create quantity vector
quantity <-  25

# Set the is-else statement
if (quantity > 20) {
   print('You sold a lot!')
} else {
   print('Not enought for today')  
}
```

```
## [1] "You sold a lot!"
```

*note*: Make sure you correctly write the indentations. Code with many conditions can become unreadable when the indentations are not in correct position. 

### Multiple conditions statement

You can customize further the control level with the `else if` statement. With `else if`, it is possible to add as many conditions as you want. The syntax is:

```
if (condition1) {
  expr1
} else if (condition2) {
  expr2
} else if (condition3) {
  expr3
} else {
  expr4
}
```

You are interested to know if you sold quantities between 20 and 30. If you do, then the statement return `Average day`, if it is above, it prints `What a great day!`, otherwise `Not enought for today`.

You can try to change the amount of quantity.


```r
# Create vector quantiy
quantity <-  10

# Create multiple condition statement
if (quantity < 20) {
  print('Not enough for today')
} else if (quantity > 20  & quantity <= 30) {
  print('Average day')
} else {
  print('What a great day!')
}
```

```
## [1] "Not enough for today"
```

### Nested if else

The `else if` statement is a clear way to understand the conditions in our code. It basically avoids to embbed many `if` inside the statement. Below, you add an easy example. VAT has different rate according to the product purchased. Imagine you have three different kind of products with different VAT applied:

| Categorie | Products                         | VAT |
|-----------|----------------------------------|-----|
| A         | Book, magasine, newspaper, etc.. | 8%  |
| B         | Vegetable, meat, beverage, etc.. | 10% |
| C         | Tee-shirt, jean, pant, etc..     | 20% |

You can write a chain to apply the correct VAT rate to the product a customer bought.


```r
categorie <- 'A'
price <- 10

if (categorie =='A'){
  cat('A vat rate of 8% is applied.', 'The total price is',price *1.08)  
} else if (categorie =='B'){
    cat('A vat rate of 10% is applied.', 'The total price is',price *1.10)  
} else {
    cat('A vat rate of 20% is applied.', 'The total price is',price *1.20)  
}
```

```
## A vat rate of 8% is applied. The total price is 10.8
```

### For loop

A `for` loop is very valuable when you need to iterate over a list of elements or a range of numbers. Loop can be used to iterate over a list, data frame, vector, matrix or any other object. The braces and square bracket are compulsory.

The basic syntax is the following:

```
For (i in vector) {
	Exp
}
```
Here, R will loop over all the `i` in `vector` and do the computation written inside the `Exp. 

<img src="/project/conditional-statement_files/4.png" width="75%" style="display: block; margin: auto;" />

### Examples

**Example 1**  

We iterate over all the elements of a vector and print the current value.


```r
# Create fruit vector
fruit <- c('Apple', 'Orange', 'Passion fruit', 'Banana')

# Create the for statement
for ( i in fruit){
  print(i)
}
```

```
## [1] "Apple"
## [1] "Orange"
## [1] "Passion fruit"
## [1] "Banana"
```

**Example 2**

Creates a non-linear function by using the polynomial of $x$ between 1 and 4 and you store it in a list.


```r
# Create an empty list
list <- c()
# Create a for statement to populate the list
for (i in seq(1, 4, by=1)) {
  list[[i]] <- i*i
}
print(list)
```

```
## [1]  1  4  9 16
```

The `for` loop is very valuable for machine learning task. After you trained a model, you need to regularise the model to avoid over-fitting. Regularization is a very tedious task because you need to find the value that minimises the loss function. To help us detect those value, you can make use of a `for` loop to iterate over a range of values and define the best candidate. 

### Loop over a list

Looping over a list is just as easy and convenient as looping over a vector.

Let's see an example


```r
# Create a list with many three vectors
fruit <- list(Basket = c('Apple', 'Orange', 'Passion fruit', 'Banana'), Money = c(10, 12, 15), purchase = FALSE)

for (p  in fruit) {
  print(p)
}
```

```
## [1] "Apple"         "Orange"        "Passion fruit" "Banana"       
## [1] 10 12 15
## [1] FALSE
```

### Loop over a matrix

A matrix has 2-dimension, rows and columns. To iterate over a matrix, you have to define two `for` loop, namely one for the rows and another for the column. 


```r
# Create a matrix
mat <- matrix(data = seq(10, 20, by=1), nrow = 6, ncol =2)

# define the double for loop
for (i in 1:nrow(mat)) {
  for (j in 1:ncol(mat)) {
  print(paste("Row", i, "and column", j, "have values of", mat[i,j]))
  }
}
```

```
## [1] "Row 1 and column 1 have values of 10"
## [1] "Row 1 and column 2 have values of 16"
## [1] "Row 2 and column 1 have values of 11"
## [1] "Row 2 and column 2 have values of 17"
## [1] "Row 3 and column 1 have values of 12"
## [1] "Row 3 and column 2 have values of 18"
## [1] "Row 4 and column 1 have values of 13"
## [1] "Row 4 and column 2 have values of 19"
## [1] "Row 5 and column 1 have values of 14"
## [1] "Row 5 and column 2 have values of 20"
## [1] "Row 6 and column 1 have values of 15"
## [1] "Row 6 and column 2 have values of 10"
```

### While loop
A loop is a statement that keeps running until a condition is satisfied. The syntax for a while loop is the following:

```
while (condition) {
	Exp
	}
```

<img src="/project/conditional-statement_files/5.png" width="55%" style="display: block; margin: auto;" />

*note*: Remember to write a closing condition at some point otherwise the loop will go on indefinitely.

**Example 1**

Let's go through a very simple example to understand the concept of while loop. You will create a loop and after each run add 1 to the stored variable. You need to close the loop, therefore we explicitely tells R to stop looping when the variable reached 10.

*note*: If you want to see current loop value, you need to wrap the variable inside the function `print()`. 


```r
# Create a variable with value 1
begin <- 1
# Create the loop
while (begin <= 10){
  # See which we are
  cat('This is loop number', begin)
  # add 1 to the variable begin after each loop
  begin <- begin +1
  print(begin)
}
```

```
## This is loop number 1[1] 2
## This is loop number 2[1] 3
## This is loop number 3[1] 4
## This is loop number 4[1] 5
## This is loop number 5[1] 6
## This is loop number 6[1] 7
## This is loop number 7[1] 8
## This is loop number 8[1] 9
## This is loop number 9[1] 10
## This is loop number 10[1] 11
```

**Expample 2**

You bought a stock at price of 50 dollars. If the price goes below 45, we want to short it. Otherwise, we keep it in our portfolio. The price can fluctuate between -10 to +10 around 50 after each loop. You can write the code as follow:


```r
set.seed(123)
# Set variable stock and price
stock <- 50
price <- 50
# Loop variable counts the number of loops 
loop <- 1
# Set the while statement
while (price > 45){
  # Create a random price between 40 and 60
  price <- stock + sample(-10:10, 1)
  # Count the number of loop
  loop = loop +1 
  # Print the number of loop
  print(loop)
}
```

```
## [1] 2
## [1] 3
## [1] 4
## [1] 5
## [1] 6
## [1] 7
```

```r
cat('it took', loop, 'loop before we short the price. The lowest price is', price)
```

```
## it took 7 loop before we short the price. The lowest price is 40
```
