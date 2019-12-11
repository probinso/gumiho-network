NOT CURRENTLY WORKING

# Gumiho-Network

This is a network for unsupervised anomaly detection inspired by the neyman pearson lemma.

This implements the network found [here](https://arxiv.org/pdf/1810.09133.pdf). The implementation is modular, setup to process batches, and adapatable to different encoding/decoding networks. The final network exists in `gumiho_network.py`

# Notes

There are admittedly no comments in this project, but it survided 5 re-writes in persuit of clear implementation. I suggest readers to ramp up code by complexity of network. A basic understanding of Variational Auto Encoders is suggested to read and understand this code. Additionally, the [research paper](https://arxiv.org/pdf/1810.09133.pdf) this work is based on is a good resource for understanding the final product.

It is easiest to ramp onto this technology by following these files by complexity as listed below

# Files Description

- `ae.py` is an autoencoder and data loading functions
- `vae.py` is supporting abstractions to variational autoencoder
- `gumiho.py` is a VAE network with support for multiple tails
- `gmm.py` is the gaussian mixture model
- `gumiho_descriminator.py` contains a descriminating generator and the complete network

All greek letters in the paper are used directly. I use the terms False Atypical Rate and True Atypical Rate instead of `FPR` and `TPR`

I also use the words `typical` and `atypical` instead of `normal` and `anomaly`. This is to avoid main collision with gaussian.

The `Encoder` and `Decoder` networks are described in `ae.py`. They are super simple neural networks.

# Setup

```bash
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

# Run

any of the given python files can be run to test that individual component

```bash
source venv/bin/activate
python gumiho_descriminator.py
```
