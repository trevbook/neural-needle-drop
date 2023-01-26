### Dec. 22nd, 2022
- Started the project
- Got tripped up on the installation of Whisper for a bit. Seems like pip-installing the repo is NOT the only thing you 
have to do. I'd also recommend uninstalling the bundled `torch`, `torchvision`, and `torchaudio` libraries and manually
installing them through the pip-command listed on [the Pytorch website](https://pytorch.org/get-started/locally/#start-locally).
My command was: 

```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### Holiday 2023
- Throughout the week of Dec. 25th - Dec. 31st 2023, I basically kickstarted this project. Wrote a bunch 
of Jupyter notebooks that scraped all of the Fantano videos, transcribed them, and embedded the transcriptions.
Totally forgot that I made this DEVLOG file, so I figured that I ought to pick it back up in the New Year. 

### Jan. 8th, 2023
- I spent some time creating methods to enrich the data
- I spent some time creating a prototype for the Search functionality in Dash 

### Jan. 12th, 2023
- I started learning about MySQL, and played around with setting up a dummy server (thanks to some help from my favorite robot friend, ChatGPT!) The following represents my progress by the end of my experimentation: 

![](./devlog-assets/mySQL%20day%20one%20experiments.png)

- I made the `MySQL Experimentation` notebook - this contains some code for querying my new MySQL database. 

- I also started exploring VSCode a lot more. I followed [this tutorial](https://vscode.rocks/minimal-ui/) to add a "zen mode" keyboard shortcut - CTRL+T, CTRL+Z. It'll be helpful when I want to JUST be looking at a Jupyter Notebook in the window - sorta like Zen mode, but windowed. 

### Jan. 16th, 2023
- I started playing around with [Pinecone](https://www.pinecone.io/), which is a vector database. My experiments are within the `Pinecone Experimentation` notebook.  
- I also added a new table to my MySQL database: `embeddings`. This table - as the name would suggest - contains embeddings for each of the videos. 


### Jan. 18th, 2023
- I made the `transcriptions` table within the MySQL database
- I finally finished uploading all of my vectors to Pinecone, and then wrote the `Embedding Search with MySQL and Pinecone` notebook - this combined my work with MySQL and Pinecone to make a *much* faster querying process
- I started the `results_prototype.py` Dash app to prototype how to display the results from a Dash video 
- Once I'd gotten a fairly rudimentary version of the Search Results look and feel, I moved the component into the newly-created `custom_components.py`, and then integrated it into the Search prototype. 
  - This marks the first time I've got a fully-fledged working prototype of this tool using *two* different database systems! 


### Jan. 22nd, 2023
- Spent some time trying to benchmark and speed up the neural search. Got things down to ~3sec for total search time (down from 13 seconds; one of these things was just a leftover "sleep for 5 seconds" statement, so I really dropped it from 8sec --> 3sec)


### Jan. 23rd, 2023
- Creating the `enriched_video_details` table within the MySQL database
- Incorporated filters into the Search App Prototype 


### Jan. 24th, 2023
- Started investigating the prospect of setting up a Solr server to run basic keyword queries on my data. I'm experimenting with starting this Solr server within a Docker container; if all goes well, I'll eventually move this to an EC2 instance or something (so it can be accessed from the cloud)
  - A lot of this was done w/ ChatGPT, but I also used the ["Solr in Docker" documentation](https://solr.apache.org/guide/solr/latest/deployment-guide/solr-in-docker.html) to guide me! 
- After toying around with the Solr server / Docker container for a bit, I made the **`Experimenting with Solr.ipynb`** notebook to test that I could properly configure & query from my core. There's a lot of work to be done here, but it seems to be working in a rudimentary way, which is solid! 
- I also started the `viewing_prototype.py` Dash app, which tries to place the transcript under the video (and allows you to click around through the video). There's still a *decent* amount to figure out, but... I think it's a decent start!  


![](./devlog-assets/viewing%20prototype.png)


### Jan. 25th, 2023
- Spent a while throughout the day designing a custom Dash component that'd allow me to get some nice hover effects / styling on the transcript segments (so as to avoid everything being buttons...) I eventually settled on something, and then published it to [a GitLab repo (called "transcription-display")](https://gitlab.com/custom-dash-components/transcript-display)
- I edited the Search App prototype to incorporate the Transcription Display, and got something pretty solid working! I think this is honestly a solid MVP - it's pretty much the exact app I'd envisioned when I'd first thought of things about a month ago. There are plenty of additional features to add, but I should be excited that I got something working this far. 

