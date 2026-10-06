# Langevin Flow Maps

Accompanying code repository for [Langevin Flow Maps: Efficient Molecular Dynamics and Transition Path Sampling](https://arxiv.org/abs/2610.05998).

The training and evaluation code for the single-molecule experiments is provided at `experiments/train.py` and `experiments/eval.py`. The training code for the transferable experiment is provided at `experiments/train_transferable.py`.

## Installation

To install the dependencies, run `uv sync`. The project requires Python 3.10+ and includes JAX with CUDA 12 support for NVIDIA GPUs.

## Datasets

### Single-molecule

We use the MD17 organic molecule datasets from [SGDML](https://www.sgdml.org/#datasets) and the Alanine Dipeptide dataset form [Smooth Normalizing Flows](https://github.com/noegroup/smooth_normalizing_flows). The single-molecule trainer downloads the selected dataset automatically into `data/`. Supported datasets are `aspirin`, `ethanol`, `naphthalene`, `salicylic`, `paracetamol` and `aldp`. For example

```bash
uv run python -c 'from experiments.train import main; main(dataset="aspirin", temperature=500.0)'
uv run python -c 'from experiments.train import main; main(dataset="aldp", temperature=300.0)'
```

### Many Peptides

We use the [Many Peptides](https://huggingface.co/datasets/transferable-samplers/many-peptides-md) dataset for the transferable experiments. The trainer downloads force shards from [Many Peptides Forces](https://huggingface.co/datasets/niklastr/many_peptides_forces) as they are needed, and training PDBs from [Many Peptides](https://huggingface.co/datasets/transferable-samplers/many-peptides-md).

```bash
uv run python experiments/train_transferable.py
```
