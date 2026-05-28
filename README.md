# meta-dynamical-ssm

Author code repository for:

**Meta-dynamical state space models for integrative neural data analysis**
Ayesha Vermani, Josue Nassar, Hyungju Jeon, Matthew Dowling, Il Memming Park
*The Thirteenth International Conference on Learning Representations (ICLR), 2025*
[OpenReview](https://openreview.net/forum?id=SRpq5OBpED)

From the [CATNIP Lab](https://catniplab.github.io).

## Overview

Neural recordings from similar tasks share latent structure, but most
state-space models are designed to fit a single dataset and cannot account
for statistical heterogeneity across recordings (different observation
dimensionalities, noise levels, animals, sessions). This work proposes a
meta-learning approach that captures variability across recordings on a
low-dimensional manifold parametrizing a family of related dynamics. The
shared structure enables few-shot reconstruction and forecasting of latent
dynamics given new recordings. We demonstrate the approach on synthetic
dynamical systems and on motor cortex recordings during different arm
reaching tasks.

The implementation is a PyTorch meta-learning state-space model that shares
a transition function across datasets, with per-dataset readin/likelihood
adapters, a per-dataset embedding `e`, and a hypernetwork that maps `e` to
low-rank (LoRA-style) deltas adapting the shared MLP transition.

## Citation

```bibtex
@inproceedings{Vermani2025-tb,
  title     = {Meta-dynamical state space models for integrative neural data analysis},
  author    = {Vermani, Ayesha and Nassar, Josue and Jeon, Hyungju and Dowling, Matthew and Park, Il Memming},
  booktitle = {The Thirteenth International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=SRpq5OBpED}
}
```
