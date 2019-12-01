# Gumiho-Network

This is an implementation of the network found [here](https://arxiv.org/pdf/1810.09133.pdf). The implementation is modular, setup to process batches, and adapatable to different encoding/decoding networks.

# Setup

```bash
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

# Files Description

- `gmm.py` is the gaussian mixture model
- `ae.py` is an autoencoder and data loading functions
- `vae.py` is supporting abstractions to variational autoencoder
- `gumiho.py` is a VAE network with support for multiple tails
- `gumiho_descriminator.py` contains a descriminating generator and the complete network

All greek letters in the paper are used directly. I use the terms False Atypical Rate and True Atypical Rate instead of `FPR` and `TPR`

I also use the words `typical` and `atypical` instead of `normal` and `anomaly`. This is to avoid main collision with gaussian.

The `Encoder` and `Decoder` networks are described in `ae.py`. They are super simple neural networks.

# Run

any of the given python files can be run to test that individual component

```bash
source venv/bin/activate
python gumiho_descriminator.py
```
