import json
import shutil
import tarfile
import tempfile
import urllib.request
from os import PathLike
from pathlib import Path

import ase
import h5py
import numpy as np
from ase.data import atomic_numbers

from langevin_flow_maps.datasets.common import (
    Dataset,
    DatasetSplit,
    split_dataset,
    unit_map,
)


def download_aldp(
    data_path: str | PathLike[str] = "data/aldp/aldp.npz",
    url="https://ftp.mi.fu-berlin.de/pub/cmb-data/bgmol/datasets/minipeptides/AImplicitUnconstrained.tgz",
):
    output = Path(data_path)
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / "AImplicitUnconstrained.tgz"
        trajectory_path = temporary / "traj0.h5"
        print("Downloading AImplicitUnconstrained...", flush=True)
        urllib.request.urlretrieve(url, archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            member = next(
                member
                for member in archive
                if member.name.endswith("AImplicitUnconstrained/traj0.h5")
            )
            with (
                archive.extractfile(member) as source,
                trajectory_path.open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
        print("Converting to aldp.npz...", flush=True)
        with h5py.File(trajectory_path, "r") as data:
            topology = json.loads(data["topology"][0])
            atoms = sorted(
                [
                    atom
                    for chain in topology["chains"]
                    for residue in chain["residues"]
                    for atom in residue["atoms"]
                ],
                key=lambda atom: atom["index"],
            )
            np.savez_compressed(
                temporary / "aldp.npz",
                R=data["coordinates"][:] * 10.0,
                F=data["forces"][:] / 41.84,
                E=data["potentialEnergy"][:] / 4.184,
                z=np.array([atomic_numbers[atom["element"]] for atom in atoms]),
            )
        (temporary / "aldp.npz").replace(output)
    print(f"Saved {output}", flush=True)


def process_aldp(
    data_path: str | PathLike[str] = "data/aldp/aldp.npz",
    center_positions: bool = True,
) -> Dataset:
    with np.load(data_path) as data:
        positions = data["R"] * unit_map["Ang"]
        forces = data["F"] * unit_map["kcal/mol/Ang"]
        energies = data["E"].reshape(-1, 1) * unit_map["kcal/mol"]
        atomic_numbers = data["z"].astype(int)

    if center_positions:
        positions = positions - positions.mean(axis=1, keepdims=True)

    molecule = ase.Atoms(atomic_numbers)
    masses = molecule.get_masses().reshape(1, -1, 1)
    atomic_numbers = molecule.get_atomic_numbers().reshape(1, -1, 1)

    return Dataset(positions, forces, energies, masses, atomic_numbers)


def load_aldp(
    data_path: str | PathLike[str] = "data/aldp/aldp.npz",
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    center_positions: bool = True,
) -> DatasetSplit:
    download_aldp(data_path)
    data = process_aldp(data_path, center_positions=center_positions)
    return split_dataset(data, splits=splits, seed=seed)
