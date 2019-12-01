# Questions

- Should we be including `KL_loss` in `_phase_two`?
- It looks like the TPR and FPR are saturated. Is there an obvious reason for this?
- Does the network correctly reflect the training order described in the paper?
- Does the training order described in the paper make sense?

# Install

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
- `gumiho_descriminator.py` is the complete network

All greek letters in the paper are used directly. I use the terms False Atypical Rate and True Atypical Rate instead of `FPR` and `TPR`

I also use the words `typical` and `atypical` instead of `normal` and `anomaly`. This is to avoid main collision with gaussian.

The `Encoder` and `Decoder` networks are described in `ae.py`. They are super simple neural networks.

# Run

any of the given python files can be run to test that individual component

```bash
source venv/bin/activate
python gumiho_descriminator.py
```
