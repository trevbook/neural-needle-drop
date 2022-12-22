### Dec. 22nd, 2022
- Started the project
- Got tripped up on the installation of Whisper for a bit. Seems like pip-installing the repo is NOT the only thing you 
have to do. I'd also recommend uninstalling the bundled `torch`, `torchvision`, and `torchaudio` libraries and manually
installing them through the pip-command listed on [the Pytorch website](https://pytorch.org/get-started/locally/#start-locally).
My command was: 

```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```