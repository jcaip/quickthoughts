# QuickThought
pytorch reimplementation of quickthoughts paper: https://arxiv.org/pdf/1803.02893.pdf

### Setup

First you'll need to pull down the bookcorpus submodule and follow the instructions there to get `all.txt`.
Next move `all.txt` to the root folder and then run `make data`, which will process the data into `cleaned.txt`.

Next grab the pretrained word vectors by running `make pull-vec`.

You'll be able to train with `make train`.

The model will pull from a config file, which can be edited to tune parameters.
