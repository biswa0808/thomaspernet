In this tutorial, you will learn how to create a website using Rstudio. 

Before we start, you need the following:

1. Rstudio
2. GitHub account
3. GitHub Destock installed



**Step 1)** Install Hugo Academic Template

Deploying in Netlify through GitHub is smooth. Yihui and Amber give some [beginner instructions](https://bookdown.org/yihui/blogdown/deployment.html), but Netlify is so easy, I recommend that you skip dragging your `public` folder in and instead [automate the process through GitHub](https://bookdown.org/yihui/blogdown/netlify.html#netlify).

Go to this [URL](https://app.netlify.com/start/deploy?repository=https://github.com/sourcethemes/academic-kickstart) and connect to GitHub. 



![1](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/1.png)

After you connect to GitHub, a new repository named `academic-kickstart` is created. Click on Save & Deploy. 



![2](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/2.png)



Netlify will deploy your site and assign you a random subdomain name of the form `blissful-mcclintock-442c04.netlify.com`.  You should know that you can change this; I changed mine to `apreshill.netlify.com`.

![3](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/3.png)

Back to your repository, you should see lots of new files. 

![4](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/4.png)

![5](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/5.png)



**Step 2)** Clone the reposityory in your local machine

You need to clone the repo on your local machine. I recommend you to use GitHub Destock, it is so easy to use, it will avoid you some unfortunate headache. 



![6](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/6.png)

Selet the name of the repository you want to clone but also the path you want to store the repo. Mine was `/Users/Thomas/Dropbox/Learning/GitHub/project/academic-kickstart`

![7](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/7.png)



You can check that the repository is effectvely cloned in your local machine in the dedicated path. 

![8](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/8.png)

There are three important folders:



- Content: Contains the posts and articles
- Static: Contains the images
- `config.toml`: Configure the static website



![9](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/9.png)



You can actually visualize your website in the URL provided by Netlify



![10](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/10.png)

**Step 3)** Change the URL

Let's start by changing the URL. Go to domain setting and edit the site name

![11](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/11.png)



You can rename the site `myfirstsitethatrock`. Save and close the windows

![12](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/12.png)



Now that your website named is changed and you have cloned the repo, you can start to edit the website. 

We recommend you to set a project directory path inside Rstudio. It will make it easier to create and post articles in the website. 



### **Step 4** Change the `toml` file

Open Rstudio and create a project inside the subdirectories when you cloned the github repo

Relevant reading:

- [`blogdown` book chapter on the RStudio IDE](https://bookdown.org/yihui/blogdown/rstudio-ide.html)

Addins: use them- you won’t need the `blogdown` library loaded in the console if you use the Addins. My workflow in RStudio at this point (again, just viewing locally because we haven’t deployed yet) works best like this:

1. Open the RStudio project for the site
2. Use the **Serve Site** add-in (only once due to the magic of *LiveReload*)
3. View site in the RStudio viewer pane, and open in a new browser window while I work
4. Select existing files to edit using the file pane in RStudio
5. After making changes, click the save button (don’t `knit`!)- the console will reload, the viewer pane will update, and if you hit refresh in the browser your local view will also be updated
6. When happy with changes, add/commit/push changes to GitHub with GitHub Destock

Open the toml file:

![11](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/13.png)

Inside this file, you will:

- Change the title

![12](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/14.png)

- Add [Disqus](https://disqus.com/): enable comment

<div id="disqus_thread"></div>
<script>
(function() {
var d = document, s = d.createElement('script');
s.src = 'https://thomaspernet.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>

Go to Disqus and create an account

1. Create account

![15](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/15.png)



2. Add site



![16](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/16.png)



3. Configure Disqus

![17](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/17.png)



4. Add html code to comment files

https://portfolio.peter-baumgartner.net/2017/09/10/how-to-install-disqus-on-hugo/

```html
<div id="disqus_thread"></div>
<script>
(function() {
var d = document, s = d.createElement('script');
s.src = 'https://thomaspernet.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
```





![18](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/18.png)

![19](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/19.png)



- Change information

![20](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/20.png)

## Change the template

## **Title**

![21](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/21.png)



![28](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/28.png)



![24](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/24.png)

![25](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/25.png)

![26](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/26.png)

![27](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/27.png)

## **Add post**



![29](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/29.png)

![30](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/30.png)

**Insert an image**

![31](/Users/Thomas/Dropbox/Learning/GitHub/project/thomaspernet/static/project/r_blog/31.png)

