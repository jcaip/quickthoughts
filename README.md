# QuickThought
pytorch reimplementation of quickthoughts paper: https://arxiv.org/pdf/1803.02893.pdf

This is a very rough implementation atm, so feel free to reach out if you're having trouble running the code.

There's also a lot of code for some experiments I ran [here](https://jcaip.github.io/Quickthoughts/) in this repo.

### Setup

First you'll need to pull down the bookcorpus submodule and follow the instructions there to get `all.txt`.
Next move `all.txt` to the root folder and then run `make data`, which will process the data into `cleaned.txt`.

Next grab the pretrained word vectors by running `make pull-vec`.

You'll be able to train with `make train`.

The model will pull from a config file, which can be edited to tune parameters.


