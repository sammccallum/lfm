import os
import urllib.request
from typing import Literal

import ase
import numpy as np

from .common import Dataset, DatasetSplit, split_dataset, unit_map

DatasetName = Literal[
    "aspirin", "ethanol", "naphthalene", "salicylic", "paracetamol"
]

data_dir = "data/md17/"


remote_stem = {
    "aspirin": "md17_aspirin",
    "ethanol": "md17_ethanol",
    "naphthalene": "md17_naphthalene",
    "salicylic": "md17_salicylic",
    "paracetamol": "paracetamol_dft",
}


def download_md17(dataset: DatasetName):
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, dataset + ".npz")

    if os.path.exists(data_path):
        return

    url = (
        "https://sgdml.org/secure_proxy.php?file=data/npz/"
        + remote_stem[dataset]
        + ".npz"
    )
    urllib.request.urlretrieve(url, data_path)


def process_md17(
    dataset: DatasetName,
    pos_unit="Ang",
    force_unit="kcal/mol/Ang",
    energy_unit="kcal/mol",
    center_positions=True,
) -> Dataset:
    pos_unit = unit_map[pos_unit]
    force_unit = unit_map[force_unit]
    energy_unit = unit_map[energy_unit]

    data_path = os.path.join(data_dir, dataset + ".npz")
    with np.load(data_path) as data:
        positions = data["R"] * pos_unit
        forces = data["F"] * force_unit
        energies = data["E"].reshape(-1, 1) * energy_unit
        atomic_numbers = data["z"].astype(int)

    if center_positions:
        positions = positions - positions.mean(axis=1, keepdims=True)

    molecule = ase.Atoms(atomic_numbers)
    masses = molecule.get_masses().reshape(1, -1, 1)
    atomic_numbers = molecule.get_atomic_numbers().reshape(1, -1, 1)

    return Dataset(positions, forces, energies, masses, atomic_numbers)


def load_md17(
    dataset: DatasetName,
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    center_positions: bool = True,
) -> DatasetSplit:
    download_md17(dataset)
    data = process_md17(dataset, center_positions=center_positions)
    return split_dataset(data, splits=splits, seed=seed)
