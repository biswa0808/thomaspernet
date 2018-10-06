HTML table to Medium
====================

Issue with Medium
-----------------

Medium is a great place to share our story with everyone. I found myself
very enthusiastic with Medium to publish my tutorials on Machine
Learning. Medium allows the user to import any URL in a
ready-to-published format. In my case, this is exactly what I need,
publish my tutorials stored on my website to Medium. It's very easy to
do, copy the URL, write a new story and import from URL. What a gain of
time! if... the URL does not contain HTML table.

The time saved with this method turns into a concrete waste. Medium does
not render HTML table. There are two ways to overcome this issue:

-   Take a screenshot of the table and embed it as a picture

-   Use Sheetsu to convert the table into an URL

The first method is, of course, unsustainable, time-consuming and give a
push-back for the reader. The second method is interesting for one
reason, the table is beautifully rendered in the Medium story.

Below is a screenshot from one of my story including a table made by
Sheetsu.

![1](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/post/table_to_medium/1.png)

Sheetsu offers an elegant solution in four steps.

1.  Create a CSV file with your data

2.  Import the CSV file in Google Spreadsheet. Make sure to share the
    spreadsheet with everyone

3.  Import the URL to Sheetsu. Under the hood, Sheetsu will create a
    nice table for Medium

4.  Go back to your Medium's story and paste the link anywhere you want



![2](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/post/table_to_medium/2.png)

There is nothing new in this story so far. There is plenty of
documentation on Google to explain this process. One shortcoming with
the overall framework is all the manual steps. Imagine you need to
import many links to Medium with all of them include a couple of tables.
It becomes a hugely time-consuming and painful. In my case, I prefer to
grab a coffee or learn new stuff rather than doing all of these.
Although, I really want to publish my tutorials on Medium. What a big
deal!

Python program
--------------

I read an article one day about a programmer. The guy made a point, when
he needs to perform twice the same thing, he wrote a code. I adopted a
similar philosophy with a slight change. When something takes more than
two repetitive steps, I write a code.

Back to our Table to Medium example, this is clear there are more than
two repetitive steps. In this story, I'll share with you the program I
wrote to do everything automatically, giving you all the time you need
to read other stories on Medium or go enjoy a coffee on a terrace while
your machine is working.

A reminder on the fly, the program is fully automatic for MacOs user.
The reason is, I'm a MacOs user and there is an awesome guy that wrote a
program to publish markdown file to Medium. I haven't browse for such
program in Windows yet.

### Table to Medium

In brief word, the program uses a valid URL, search for tables, creates
one or more Google Spreadsheet, import them to Sheetsu, change the HTML
file, save it as markdown format and publish it to Medium.

All of these steps above are done automatically.

**Requirement**

There are few requirements to use the program:

-   Have Python installed on the machine

-   Have a Google account

-   Have Selenium installed

If you don't have Selenium installed or you don't want to install it,
which I understand, you can still use the program, but you need to
perform one step manually. The reason I use Selenium is, Sheetsu API
does not render the table in Medium. Their API saves the table in a JSON
format, which is not read by Medium.

Let's begin, you can install the program with this command. We will
proceed in three steps:

1.  Prepare the files

2.  Prepare Table-to-markdown (optional, only for MacOS user)

3.  Launch the codes

Step two is optional. It is meant to use the program I mentioned before,
take a markdown file and publish it to Medium.

**Step 1: Prepare the files**

You need to have a valid URL and the HTML file saved locally. You can
save the website as HTML without style.

For instance, all my tutorials are stored locally.

![3](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/post/table_to_medium/3.png)

If I want to publish the first tutorial, I open the file and save it in
HTML without style;

![4](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/post/table_to_medium/4.png)

**Step 2: Prepare Table-to-markdown**

This step helps you to get your token in Medium. You will need it later
to publish your story. Go to settings in Medium and copy somewhere your
token.

![5](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/post/table_to_medium/5.png)

**Step 3: launch the code**

The program is divided into four part

![6](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/post/table_to_medium/6.png)

**Step 1: extract and populate Google Spreadsheet**

In this steps, we will declare the variables and use the function
`extract_table_url` to extract and populate Google spreadsheet.

We need to declare two variables:

-   The Url
    `https://thomaspernet.netlify.com/tensorflow/what-is-machine-learning/`

-   The path where we stored our Google authorization. You can follow
    this tutorial to get your Access Google APIs
    `/users/Thomas/Oath_Docker_Gcloud/Google_auth/`

    import import_medium as md
    
    url = "https://thomaspernet.netlify.com/tensorflow/what-is-machine-learning/"
    path = '/users/Thomas/Oath_Docker_Gcloud/Google_auth/'

Then we declare the variable

```python
make_table = md.Spreadsheet(url = url, path = path)
```

If you want to know how many tables will be extracted, you can use this
code

```pyto
print(make_table.count_table())
```

The result is 2.

You should be able to run `extract_table_url` and create the
spreadsheets.

```python
preparation_selenium = make_table.extract_table_url()
```

After you run the code, you should see a couple of useful pieces of
information:

-   The link to the Google spreadsheet

-   The ID of the spreadsheet

![7](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/post/table_to_medium/7.png)

**Step 2: Automate table creation in Sheetsu**

If you have Selenium installed in your machine, you can do this step. If
you don't, you need to manually perform the creation in Sheetsu. Just
copy the URL of the Google spreadsheet and copy them here

We need to locate where we put Selenium drivers. In my case, it is
`/Users/Thomas/Selenium/chromedriver`. Now that we know where is the
driver, we can use the function `add_to_sheetsu` to automate the job.

```python
path_sheetsu = '/Users/Thomas/Selenium/chromedriver'
list_sheetsu = [md.add_to_sheetsu(i, path_selenium = path_sheetsu) for i in preparation_selenium]
```

After you run the code, a new web browser is open. It will create as
many new tables as in the original Url. Sheetsu will provide the Url to
copy in our story.

We can use `list_sheetsu` to see the Sheetsu' Url.

```python
list_sheetsu
```

**Step 3: Change the HTML**

Make sure you have saved the HTML without style locally and copied the
path.

![8](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/post/table_to_medium/8.png)

We can define two new variables, the path of the HTML and the path where
we want to store the markdown file. Note that, the markdown file will be
used to publish on Medium.

-   Path to HTML:
    `/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/medium/02_What_is_Machine_Learning_v8.html`

-   Path the markdown:
    `/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/medium/What_is_Machine_Learning.md`

Note that, we **have to** use the extension `md`. `md` stands for
markdown.

```python
PATH = "/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/medium/02_What_is_Machine_Learning_v8.html"
title = "/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/medium/What_is_Machine_Learning.md"
```

We need to add this two variables to the class `Table_html`

```python
table_md = md.Table_html(path = PATH, title = title)
```

Finally, we make use of `bs_change_table` to create the new markdown
file.

```python
table_md.bs_change_table(list_i = list_sheetsu)   
```

The program tells us the file is ready to publish. We can have a look at
the markdown file.

![9](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/post/table_to_medium/9.png)

We are ready to publish the article on markdown.

**Step 4: Publish to Medium**

Before to publish the markdown file to Medium, make sure you have
installed this program

We need to create the command line to call in Python so that we can
publish our story.

The program `markdown-to-medium` is relatively simple, we need to call
the program, specify the path of the markdown and copy our token. With
Python, copy and paste the following code. Replace `YOUR TOKEN` with the
token provided by Medium (i.e, step 2)

```python
import os
Medium ="markdown-to-medium "+title + " --token=YOUR TOKEN"
os.system(Medium)
print("done exporting!")
```

We are redirected to Medium with our article as a draft.

The full program

```python
import import_medium as md

### Change Url and path
url = "https://thomaspernet.netlify.com/tensorflow/what-is-machine-learning/"
path = '/users/Thomas/Oath_Docker_Gcloud/Google_auth/'

print("Step 1: extract and populate Google Spreadsheet")
make_table = md.Spreadsheet(url = url, path = path)
preparation_selenium = make_table.extract_table_url()

### Change path_selenium
print("Step 1: extract and populate Google Spreadsheet")
path_selenium = '/Users/Thomas/Selenium/chromedriver'

list_sheetsu = [md.add_to_sheetsu(i, path_selenium = path_selenium) for i in preparation_selenium]

### Change PATH and title
print("Step 3: Change the HTML")
PATH = "/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/medium/02_What_is_Machine_Learning_v8.html"
title = "/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/medium/What_is_Machine_Learning.md"

table_md = md.Table_html(path = PATH, title = title)
table_md.bs_change_table(list_i = list_sheetsu)    

### Change token
print("Step 4: Publish to Medium")
import os
Medium ="markdown-to-medium "+title + " --token=YOUR_TOKEN"
os.system(Medium)	
print("done exporting!")
```

**Bonus**

We can delete unecesarry files like the HTML or the spreadsheet in
Google. For that, you need to copy the spreadsheet ID. There are in the
variables `preparation_selenium` . After we have copy them, we can them
we an use `remove_table`

```python
preparation_selenium
```

We get:

    ['https://docs.google.com/spreadsheets/d/1HGpBhsERL0VhlVf4qwWTAyZBXZC2wxluFzPP07F8bdw/edit?usp=sharing', 'https://docs.google.com/spreadsheets/d/1JodhV4nMIpLPbar15MAekIhMIpPjpilRvfbTI5Br13A/edit?usp=sharing']

The ID is `1HGpBhsERL0VhlVf4qwWTAyZBXZC2wxluFzPP07F8bdw` and
`1JodhV4nMIpLPbar15MAekIhMIpPjpilRvfbTI5Br13A`. You can store them in a
list

```python
list_id = ["1HGpBhsERL0VhlVf4qwWTAyZBXZC2wxluFzPP07F8bdw","1JodhV4nMIpLPbar15MAekIhMIpPjpilRvfbTI5Br13A"]
```

At last, we can delete them very easily.

```python
os.remove(PATH)
[test.remove_table(i) for i in list_id]
```

You should see a message confirming the spreadsheets have been deleted.

    file 1HGpBhsERL0VhlVf4qwWTAyZBXZC2wxluFzPP07F8bdw deleted
    file 1JodhV4nMIpLPbar15MAekIhMIpPjpilRvfbTI5Br13A deleted

### Details program

The details of the program is listed below

![10](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/post/table_to_medium/10.png)
