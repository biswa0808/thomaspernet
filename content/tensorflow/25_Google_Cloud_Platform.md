---
title: Google Cloud platform
author: Thomas
date: '2018-09-19'
slug: google
categories: []
tags:
  - extra
header:
    caption: ''
    image: ''

---

<style>
body {
text-align: justify}
</style>

Google Cloud Platform
=====================

Google started to make public its collection of computing resources in
2008. Google's suite provides services in cloud computing, data storage,
data analytics, and more recently artificial intelligence.

Google stores all of the data on its physical infrastructures like
computers, hard disk drive, and networking. The users of Google Cloud
Platform can access their data through virtualized resources (i.e.,
virtual machine).

Google Cloud Computing is equipped with services that will allow
collaborative work from data storage to the deployment of App.

Storage
-------

Google Cloud Storage is the perfect tool to store any data. It is
designed to integrate storage into an APP with a single unified API, it
is cheap and incorporate the latest security encryptions to protect the
data. On top of that, Google Cloud Storage cares about the environment.
Every file stored in Google Data Center minimizes the carbon emissions.

Big Data
--------

Big Data is revolutionalizing the industry by allowing a seamlessly
access to the data. Building a Big Data architecture is complicated,
expensive and time-consuming. Google Cloud Platform delivers an
end-to-end product to the users. With Google Cloud Platform you can
focus on finding insights rather than managing your infrastructure, and
you can combine **cloud-native services** with **open source tools** as
needed, both in **batch and stream mode**.

To get an insight of the data, Google has developed Big Query, a
Business Intelligence tool. It uses SQL to retrieve the data from the
server and push it to Google BigQuery to help you visualize the data as
you've never done before.

Google Cloud Console
--------------------

Case Study
----------

In this case study, you will learn how to use Google Cloud Platform to
set up a Data analysis project. The idea is to create an environment
where the analysis can access the data in real time without much effort.

In fact, during this process, you will store the data in Google Cloud
Storage. This tool is convenient to save a large dataset at a little
cost. The second step consists in transferring the data to Google Big
Query. It will convert the CSV file into a data table. To extract the
data, you can use SQL. Big Query allows the user to share not only the
dataset but also the results of a query. Once you are satisfied with the
queries, you can explore the data within Google Data Studio.

<img src="/tensorflow/25_google_cloud/image1.png">{width="5.833333333333333in"
height="3.533333333333333in"}

You will proceed as follow:

-   Step 1) Add the data to the cloud using the `gsutil` tool

-   Step 2) Create a dataset in Big Query

-   Step 3) Give privileges to one member of the project (pending)

-   Step 4) Create a Query to retrieve the data

-   Step 5) Open Google Data Studio

Before to go through the steps, go to this URL and download the dataset
in CSV format. This is the Titanic dataset. You will use this dataset
during the tutorial.

Note that, we assume you have already an account set up, and you have
installed `gsutil` on your machine. If `gsutil` is not installed, please
go to this URL and install the package.

### Step 1) Add the data to the cloud using the `gsutil` tool

When you connect to Google Cloud Platform, you land on this page. There
are lots of information. You can check Google documentation for further
details. You can see Google created a default project for you,
`My Project`. Google sets a new workflow for any new project. Keep in
mind that everything will be done inside this project name. A project
name is given an actual ID. To keep track of changes, you need to use
this ID. If you want to change to a new project, you need to create a
new one. A unique ID will be provided.

For this tutorial, you will use the project named `My Project`.

<img src="/tensorflow/25_google_cloud/image2.png">{width="5.833333333333333in"
height="3.2673261154855644in"}

You need to find Cloud Storage. It is located in the drop-down menu in
the left corner.

<img src="/tensorflow/25_google_cloud/image3.png">{width="4.8811187664041995in"
height="6.559440069991251in"}

The first step you need to do is to create a *bucket*. This bucket will
contain any data you need for your project. To give you an example,
imagine you are working on a non-dynamic data flow (i.e., there is no
data update with time). The data is primarily stored in ZIP format.
After the unzipping the folder, you get the CSV file. It can be disk
consuming. One way is to create a bucket where you store the ZIP file
and the CSV file.

Click on `Create bucket.
`

<img src="/tensorflow/25_google_cloud/image4.png">{width="5.833333333333333in"
height="0.5335356517935258in"}

To create a bucket, you need to add the following items:

-   A name: tutorialgc

-   Storage class: Regional

-   Location: us-west2

Note that you choose the storage class and regional with a good tradeoff
between cost and performances. Keep in mind that during the first year,
Google gives you \$300 to spend and lots of free space.

<img src="/tensorflow/25_google_cloud/image5.png">{width="5.833333333333333in"
height="4.249577865266842in"}

Now that the bucket is created, you can add as many files as you want.
In my opinion, it is more convenient to add data to the bucket by using
the terminal/command line.

First, make sure you are in the right directory. In the example below,
the CSV file `titanic_train` is stored locally on the folder
`Dataset_gcloud`.

You need to use `gsutil` to add data to the bucket. It is
straightforward:

-   `gsutil`: call the command to add data

-   `cp titanic_train.csv`: copy the data locally

-   `gs://tutorialgc`: paste the data into the bucket `tutorial
    `

<!-- -->

    cd Dataset_gcloud/ 
    gsutil cp titanic_train.csv gs://tutorialgc

If you encounter very large CSV file, you can use the parallel composite
command to faster the transfer

    ## For big file
    gsutil -o GSUtil:parallel_composite_upload_threshold=150M cp titanic_train.csv gs://tutorialgc

If you go back to Google Cloud Storage, you can see the file is
successfully uploaded.

<img src="/tensorflow/25_google_cloud/image6.png">{width="5.833333333333333in"
height="1.660424321959755in"}

You can add multiple files at the same times. Use `*` instead of the
file names and all the CSV extension in the folder will be uploaded to
the cloud.

In the next step, you will add the CSV file in Big Query.

### Step 2) Create a dataset in Big Query

Google BigQuery is a cloud-based important data analytics web service
for processing substantial read-only data sets.

BigQuery was designed for analyzing data on the order of billions of
rows, using a SQL-like syntax. It runs on the Google Cloud Storage
infrastructure and can be accessed with a REST-oriented application
program interface.

To access Big Query, go back to the console and search for Big Query.

<img src="/tensorflow/25_google_cloud/image7.png">{width="2.9930063429571305in"
height="6.419580052493438in"}

In this workload, there are the:

-   Query Editor

-   Dataset

-   Query Viewer

<img src="/tensorflow/25_google_cloud/image8.png">{width="5.833333333333333in"
height="2.1498950131233596in"}

So far, the project does not contain the dataset. You need to transfer
the CSV file to the project.

First of all, make sure you are in the right Project ID. Where Google
created the project for you, it assigned it an ID: `valid-pagoda-132423`

If you only have one project, this step is meaningless but if you work
on different projects, make sure to point to the right project:

    gcloud config set project valid-pagoda-132423

You need to create a dataset to store the data table. You name the
dataset, `tuto`

<img src="/tensorflow/25_google_cloud/image9.png">{width="5.833333333333333in"
height="2.9359820647419075in"}

Now that everything is set, you can transfer the data. The code below is
the simplest way to move the data from the bucket to Big Query. You let
Big Query parse the data with the correct format. Not that, if you have
`NaN` values, it can cause problems.


    bq --location=US load --autodetect --source_format=CSV tuto.titanic_train gs://tutorialgc/titanic_train.csv

<img src="/tensorflow/25_google_cloud/image10.png">{width="5.833333333333333in"
height="1.5459361329833772in"}

You can browse the project and see that the data has been transferred.

<img src="/tensorflow/25_google_cloud/image11.png">{width="4.727272528433946in"
height="2.3356638232720908in"}

Click on the `Schema` button to visualize how Big Query parsed the data.

<img src="/tensorflow/25_google_cloud/image12.png">{width="4.867132545931758in"
height="4.517482502187226in"}

Copy and paste the following query into the query text area.

    SELECT survived, sex, age, fare
    FROM tuto.titanic_train
    LIMIT 5 OFFSET 0;

A green check mark icon is displayed if the query is valid. If the query
is invalid, a red exclamation point icon is displayed. If the query is
valid, the validator also shows the amount of data the query will
process when you run it. The data processed is helpful for determining
the cost of running the query.

Click **Run Query**. The query results page displays below the query
window. At the top of the query results page, the time elapsed and the
data processed by the query are presented. Below the `Query complete...`
message, a table displays the query results with a header row containing
the name of each column you selected in the query.

<img src="/tensorflow/25_google_cloud/image13.png">{width="5.833333333333333in"
height="3.420711942257218in"}

### Step 3) Give privileges to one member of the project

Sharing the data is one of the critical points of using Google Big
Query. There are two ways of providing access to the data:

-   Send a temporary URL

-   Give analyst a View/Reader access

**temporary URL**

To use a service account outside of the Google Cloud Platform (on other
platforms or premise), you must establish the identity of the service
account. Public/private key pairs will let you do that.

You can create a service account key using the GCP Console, the `gcloud`
tool, the
\[\]`serviceAccounts.keys.create()`\](https://cloud.google.com/iam/reference/rest/v1/projects.serviceAccounts.keys/create)
method, or one of the client libraries.

In the examples below, **\[SA-NAME\]** is the name of your service
account, and **\[PROJECT-ID\]** is the ID of your Google Cloud Platform
project. You can retrieve the
**\[SA-NAME\]@\[PROJECT-ID\].iam.gserviceaccount.com** string from the
Service Accounts page in the GCP Console.

In our example, the account is
`thomas2405@valid-pagoda-132423.iam.gserviceaccount.com`

    gcloud iam service-accounts keys create ~/key.json --iam-account thomas2405@valid-pagoda-132423.iam.gserviceaccount.com

After you paste the code in the terminal, a new file is added in the
root directory

<img src="/tensorflow/25_google_cloud/image14.png">{width="5.833333333333333in"
height="0.8361428258967629in"}

In here, it is `User/Thomas`

You can give temporary access to the dataset with the following code

    gsutil signurl -d 10m /Users/Thomas/key.json gs://tutorialgc/titanic_train.csv

<img src="/tensorflow/25_google_cloud/image15.png">{width="5.833333333333333in"
height="2.1385531496062993in"}

It provides the temporary URL

    https://storage.googleapis.com/tutorialgc/titanic_train.csv?x-goog-signature=9c9af851ffcf82735caf74e308fea7306295c5e0f45b702d7b74fb8bc29218a0678e490a59b4804ac96cdac1b930390520acec3a8d245234d664739c6e7414c2b8f2f8ba53b6b14bf956aa2f4237b5b71d6640211a2e3c89cbd814d89a886ee8dce2679837b82892b5834c015830ed3e3e34feda8bb477eade8dd2e772c9ab0c3f24a35f67d63c97e2b419f07f9d6153f951766cb0e3539017857286d07284716080f4aa7bb46749838fa595468ed3af7e31ccaddf4c6874fe3bb99aa3605b5ca9f5b34ee8871d5106c2a2e8535a4a47787a488462ff01f0efafd7e30214d3cc6aace8f933c90bb12825107e47891ac5c2e32674eab139f16ed1fb2c1fff0b50&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=thomas2405%40valid-pagoda-132423.iam.gserviceaccount.com%2F20180918%2Fus-west2%2Fstorage%2Fgoog4_request&x-goog-date=20180918T061332Z&x-goog-expires=600&x-goog-signedheaders=host

**Give access**

https://cloud.google.com/bigquery/docs/share-access-views

A faire

### Step 4) Create a Query to retrieve the data

Now that your project is set, you gave the privileges to your team; it
is time to get to know your data.

In this step, you will retrieve all the data from the data table. Use
can write the following query:

    SELECT *
    FROM tuto.titanic_train

<img src="/tensorflow/25_google_cloud/image16.png">{width="5.833333333333333in"
height="1.7257217847769029in"}

If your dataset is big, you don't need to query all of the data, select
the columns you need or filter the data.

-   Step 5) Open Google Data Studio

<img src="/tensorflow/25_google_cloud/image17.png">{width="4.391607611548556in"
height="0.5734262904636921in"}

To explore the data in Data Studio, click on the button
`EXPLORE IN DATA STUDIO`. Make sure you run the query before.

You will land in the Data Studio.

<img src="/tensorflow/25_google_cloud/image18.png">{width="5.833333333333333in"
height="2.220695538057743in"}

You can play around and visualize the data.

<img src="/tensorflow/25_google_cloud/image19.png">{width="5.833333333333333in"
height="2.1743667979002623in"}
