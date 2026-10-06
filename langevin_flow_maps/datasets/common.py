from typing import NamedTuple

import ase.units as units
import jax.numpy as jnp
import jax.random as jr
import numpy as np

unit_map = {
    "eV": units.eV,
    "Hartree": units.Hartree,
    "kcal/mol": units.kcal / units.mol,
    "kJ/mol": units.kJ / units.mol,
    "Ang": units.Ang,
    "nm": units.nm,
    "Bohr": units.Bohr,
    "eV/Ang": units.eV / units.Ang,
    "Hartree/Ang": units.Hartree / units.Ang,
    "Hartree/Bohr": units.Hartree / units.Bohr,
    "kcal/mol/Ang": units.kcal / units.mol / units.Ang,
    "kJ/mol/Ang": units.kJ / units.mol / units.Ang,
    "kJ/mol/nm": units.kJ / units.mol / units.nm,
}


class Dataset(NamedTuple):
    positions: np.ndarray  # (n_frames, n_atoms, 3), Ang
    forces: np.ndarray  # (n_frames, n_atoms, 3), eV/Ang
    energies: np.ndarray  # (n_frames, 1), eV
    masses: np.ndarray  # (1, n_atoms, 1), amu
    atomic_numbers: np.ndarray  # (1, n_atoms, 1)


class DatasetSplit(NamedTuple):
    train: Dataset
    val: Dataset
    test: Dataset


def split_dataset(
    data: Dataset,
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> DatasetSplit:
    n_frames = data.positions.shape[0]
    n_val = int(n_frames * splits[1])
    n_test = int(n_frames * splits[2])
    n_train = n_frames - n_val - n_test

    indices = np.random.default_rng(seed).permutation(n_frames)
    index_splits = (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )

    energy_shift = data.energies[index_splits[0]].mean()

    return DatasetSplit(
        *(
            Dataset(
                positions=data.positions[index],
                forces=data.forces[index],
                energies=data.energies[index] - energy_shift,
                masses=data.masses,
                atomic_numbers=data.atomic_numbers,
            )
            for index in index_splits
        )
    )


def dataloader(arrays, batch_size, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
