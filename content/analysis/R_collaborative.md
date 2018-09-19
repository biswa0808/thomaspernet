---
title: R and GitHub
author: Thomas
date: '2018-09-19'
slug: R
categories: []
tags:
  - intro
header:
    caption: ''
    image: ''

---

<style>
body 
</style>

R as a collaborative platform
=============================

R is a powerful tool that drives data analysis beyond boundaries.
Besides, R can embrace the powerful technology of GitHub to create a
fantastic collaborative study.

Many researchers face constraints when it comes to sharing the results
with their colleagues or peers. One standard solution implemented so far
is to exchange the files either through Dropbox, Google Drive or any
other cloud platform. The most significant challenge results in how to
track change and have consistent data over time. When the analysis
grows, the complexity becomes unsustainable to handle.

With the current technology, it is possible to create a standard
workflow for a team. One of the main advantages of collaborative work is
to avoid duplicate, mistake and other loss of files.

In this part of the textbook, you will learn how to use R with GitHub to
create a collaborative platform that can be used by any of your
colleagues, students or peer.

What is GitHub
--------------

To understand GitHub, you must first have an understanding of Git. Git
is an open-source version control system that was started by Linus
Trovalds—the same person who created Linux. Git is similar to other
version control systems—Subversion, CVS, and Mercurial to name a few.

So, Git is a version control system, but what does that mean? When
developers create something (an app, for example), they make constant
changes to the code, releasing new versions up to and after the first
official (non-beta) release.

Version control systems keep these revisions straight, storing the
modifications in a central repository. This allows developers to easily
collaborate, as they can download a new version of the software, make
changes, and upload the latest revision. Every developer can see these
new changes, download them, and contribute.

Workflow
--------

Know that you get an idea of what is GitHub, let's roll our sleeve and
prepare the workflow.

To create a collaborative platform, you need to connect Rstudio with
your GitHub account. Here is how you will proceed:

**Step 1)**: Create a GitHub repository

First of all, you need to go to GitHub to create a repository. A
repository is a folder where you can share your work with the work. You
can store anything you want. It will be accessible to anyone. Note that,
students and premium members can create private repositories.

Go to this page and create a repo. For this project, you will name the
repository name `research_paper`. You can initialize the repo with a
README. The README is a description of the content of your repo. If you
have a creative project, make sure to add a new README. It will give a
taste to the people to get to know more about the content of the
project.

<img src="/project/collaboration/image1.png" >

Now that the repository is created, you need to clone the project to
your local machine.

**Step2)**: Clone the project locally

The easiest way to clone a project when working with Rstudio is.. to use
Studio. Rstudio is equipped with a GitHub add on to ease the connection
between the repository and the local file.

So far, your repository has only one file, the `README.md`. During this
step, you will create an R project by cloning the repository on your
local machine.

Click on file, new project and Version Control.

<img src="/project/collaboration/image2.png" >

Select Git

<img src="/project/collaboration/image3.png" >

You will be asked to fill out some information about the URL and project
name

<img src="/project/collaboration/image4.png" >

You can get the URL directly from the repository.

<img src="/project/collaboration/image5.png" >

Let the Project directory name blank, Rstudio will automatically pick up
the name for you. Last, you need to select the path where you want to
clone the project.

If you go to the subdirectory where you clone the repo, you will find
one new file: `README.md`

Note that, every change you are going to make will be done in this
subdirectory. If you close Rstudio and want to work again on this
project, you need to open the project called `research_paper`.

You collaborative platform is now ready. Currently, you are the only
user; in the last step, you will learn how to add people to your
workflow.

You need to give some contents to your repository. A great way to work
with the large data file (i.e., CSV file) is to use the cloud computing
technology. Researchers are often constraints by the size of the
dataset, the flow of data between the user and the machine. To overcome
this issue, you can use Google Cloud Computing. It is a powerful cloud
computing technology that allows you to store, manage, analyze and share
your data without friction.

Ideally, the administrator of the data gives access to a subset of data
to a third party. The new user cannot change or edit the data in Google
Big Query but can write a query to retrieve the data.

When the administrator correctly sets the privilege to the data analysis
team, it is possible to connect the database to Rstudio with the library
`bigrquery`

**Step 3**): Connect to Google Big Query

In Rstudio, open a new script. You need to connect Rstsudio to your
Google Account. When using large query interactively, you'll be prompted
to authorize bigrquery in the browser. Your credentials will be cached
across sessions in `.httr-oauth`.

Go to your Google Console and copy the project ID.

<img src="/project/collaboration/image6.png" >

You can write the following code to connect with Big Query

    library(bigrquery)
    project <- "valid-pagoda-132423" 
    sql <- 'SELECT * FROM tuto.titanic_train'
    
    tb <- bq_project_query(project, sql)

The `sql` variables retrieve all the data from the dataset. This is not
recommended an approach for large dataset. You'd better prepare the data
inside the SQL editor of Big Query when possible. If not, you can import
the raw data into R using Big Query, perform all the preprocessing steps
and send the final dataset back to Google Cloud.

In this case, you only need to import the final dataset you are going to
use during the analysis.

After you run the code, you will be prompted to allow the access to
Google.

<img src="/project/collaboration/image7.png" >

and then click on allow

<img src="/project/collaboration/image8.png" >

Google precisely how many bytes you consumed to import the data.

<img src="/project/collaboration/image9.png" >

The data are now stored in the memory of your machine. You can convert
them into a data frame with the following code

    library(dplyr)
    
    titanic <- bq_table_download(tb)
    
    titanic %>% 
      select(-1) %>%
      head(10) %>%
      collect()

Note that the data won't be stored in your machine. If the dataset is
small enough, you can add it to your GitHub repository and load it each
time you want to use it.

The objective is to share the data with your colleagues. You have two
choices; the first one is to give privileges to the user on the dataset
stored in Big Query. This method is ideal for a large dataset. The
second method is to store the data in the GitHub repository.

Let's say you want your colleague to have access to the codes you wrote.
To do that, you need to push the script to the repository. Once the file
is in the repo, everyone can access it.

**Step 4**:) Add a commit

First, you need to save it locally. Then, find the icon with Git and
click on Commit.

<img src="/project/collaboration/image10.png" >

Commit in GitHub means you are about to add something to the repository.
Each commit needs to have a description. It helps the other users to
follow the change committed in the repository.

<img src="/project/collaboration/image11.png" >

The commit window in Rstudio contains different useful information. In
green, Rstudio shows you elements are new in the file you are about to
upload compared to the previous version.

To make your first commit, you need to select:

1.  What file you want to push to GitHub. In the example above, it is
    the R script named `data_analysis`.

2.  Add a commit message. This message will appear on the description of
    the file in GitHub

3.  Click on commit

<img src="/project/collaboration/image12.png" >

The job is not done yet. You need to click on `the push` to push the
data to GitHub.

<img src="/project/collaboration/image13.png" >

GitHub sent you a successful message. You can close the message box.

You can check if the file is in the GitHub page.

<img src="/project/collaboration/image14.png" >

You successfully added the R script to GitHub. Now, you are ready to
share tour work with your colleague.

This is done quickly by providing the different user the URL of your
project. They can either browse on your GitHub page to find the project
and clone it, or you can give them the URL

<img src="/project/collaboration/image15.png" >

Your colleague needs to do the first steps in Rstudio to clone the repo
in their local machine. Once it is done, they can start to work with you
on the project, each one of you will have the latest version of the
file.

Note that, you need to commit and push everytime you want to add
something new to GitHub.

**Step 5**): Update the repo

Now that you added users to your repo, you can update your local folder
directly from Rstudio with the **pull** button.

One of your colleagues added new lines of codes to the main files, and
you want to have the latest branch, then you need to pull this update

<img src="/project/collaboration/image16.png" >

Rstudio will automatically update the files in the local directory.

<img src="/project/collaboration/image17.png" >

**Step 6)**: Track changes

Last but not least, GitHub is a tool of choice to track changes. GitHub
tracks every change committed to the repository, not only for the files
containing the codes but also markdown files as well.

We added a markdown file to the GitHub repository. This file can, for
example, be your working paper.

<img src="/project/collaboration/image18.png" >

One of your colleagues is reading the files but wants to make comments,
strikethrough, changes or anything else. He/she can change the primary
data and commit a change in the repo. Your colleague can see the changes
he/she made so far.

You or the other user can track this change in the history of the
repository.

<img src="/project/collaboration/image19.png" >

If you click on the latest changes, you can see what have been changes
and add comments to it.

<img src="/project/collaboration/image20.png" >

GitHub is an incredible place to start a project. Not only for the
collaborative platform but also to share your project with the world. It
is possible to store the script in GitHub to allow anyone to visualize
the code. You can write a Jupyter notebook to let the users see the
system in action. Finally, a GitHub project can quickly be turned into a
static website, meaning you can share your project with the outside
world.
