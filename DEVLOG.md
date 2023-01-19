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
