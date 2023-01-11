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
- 
