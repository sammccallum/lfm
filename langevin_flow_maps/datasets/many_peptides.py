import io
import re
import tarfile
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from openmm import app, unit

from .common import Dataset, unit_map


class MolecularBatch(NamedTuple):
    positions: np.ndarray
    forces: np.ndarray
    atomic_numbers: np.ndarray
    masses: np.ndarray
    atom_mask: np.ndarray


class PeptideTopologies:
    def __init__(self, pdb_tar, forcefield_files):
        self.pdb_tar = Path(pdb_tar)
        self.members = {}
        with tarfile.open(self.pdb_tar, "r:") as archive:
            for member in archive:
                archive.members.clear()
                if member.isfile() and member.name.endswith(".pdb"):
                    self.members[Path(member.name).stem] = (
                        member.offset_data,
                        member.size,
                    )
        self.sequences = tuple(sorted(self.members))
        self.forcefield = app.ForceField(*forcefield_files)
        self.loaded = {}

    def __getitem__(self, sequence):
        if sequence not in self.loaded:
            offset, size = self.members[sequence]
            with self.pdb_tar.open("rb") as stream:
                stream.seek(offset)
                pdb = app.PDBFile(io.StringIO(stream.read(size).decode()))
            system = self.forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=2.0 * unit.nanometer,
                constraints=None,
            )
            atomic_numbers = np.array(
                [atom.element.atomic_number for atom in pdb.topology.atoms()],
                dtype=np.int32,
            )
            masses = np.array(
                [
                    system.getParticleMass(i).value_in_unit(unit.dalton)
                    for i in range(system.getNumParticles())
                ],
                dtype=np.float32,
            )
            self.loaded[sequence] = atomic_numbers, masses
        return self.loaded[sequence]


def iter_frames(path, topologies, sequences):
    payloads = {}
    with tarfile.open(path, "r|") as archive:
        for member in archive:
            archive.members.clear()
            if member.isdir():
                continue
            key, extension = member.name.rsplit(".", 1)
            sequence = key.rsplit("_", 1)[0]
            if sequence not in sequences:
                continue
            payloads[extension] = archive.extractfile(member).read()
            if len(payloads) == 3:
                z, masses = topologies[sequence]
                n_atoms = len(z)
                positions = (
                    np.frombuffer(payloads["bin"], dtype="<f4").reshape(1, n_atoms, 3)
                    * unit_map["nm"]
                )
                positions -= positions.mean(axis=1, keepdims=True)
                forces = (
                    np.frombuffer(payloads["frc"], dtype="<f4").reshape(1, n_atoms, 3)
                    * unit_map["kJ/mol/nm"]
                )
                energies = (
                    np.frombuffer(payloads["nrg"], dtype="<f8").reshape(1, 1)
                    * unit_map["kJ/mol"]
                )
                yield Dataset(
                    positions,
                    forces,
                    energies,
                    masses.reshape(1, n_atoms, 1),
                    z.reshape(1, n_atoms, 1),
                )
                payloads = {}


def collate_peptides(frames, max_atoms):
    size = len(frames)
    positions = np.zeros((size, max_atoms, 3), dtype=np.float32)
    forces = np.zeros_like(positions)
    atomic_numbers = np.zeros((size, max_atoms), dtype=np.int32)
    masses = np.ones((size, max_atoms), dtype=np.float32)
    atom_mask = np.zeros((size, max_atoms), dtype=bool)
    for i, frame in enumerate(frames):
        n = frame.positions.shape[1]
        positions[i, :n] = frame.positions[0]
        forces[i, :n] = frame.forces[0]
        atomic_numbers[i, :n] = frame.atomic_numbers.reshape(-1)
        masses[i, :n] = frame.masses.reshape(-1)
        atom_mask[i, :n] = True
    return MolecularBatch(positions, forces, atomic_numbers, masses, atom_mask)


class ManyPeptidesDataset:
    def __init__(self, shards, pdb_tar, *, forcefield_files, sequences=None):
        if isinstance(shards, (str, Path)):
            shards = [shards]
        self.shards = tuple(str(path) for path in shards)
        self.topologies = PeptideTopologies(pdb_tar, forcefield_files)
        if isinstance(sequences, str):
            sequences = [sequences]
        self.sequences = (
            tuple(sorted(set(sequences)))
            if sequences is not None
            else self.topologies.sequences
        )
        self.resolve_shard = Path
        self.metadata = {
            "units": "ASE",
            "energy_shift_eV": 0.0,
            "center_positions": True,
            "shards": list(self.shards),
            "pdb_tar": str(pdb_tar),
            "sequences": list(self.sequences),
            "forcefield": list(forcefield_files),
        }

    def shuffled_frames(self, shuffle_buffer, seed):
        rng = np.random.default_rng(seed)
        selected = set(self.sequences)
        while True:
            shards = list(self.shards)
            if shuffle_buffer:
                rng.shuffle(shards)
            buffer = []
            for shard in shards:
                with closing(
                    iter_frames(self.resolve_shard(shard), self.topologies, selected)
                ) as frames:
                    for frame in frames:
                        if not shuffle_buffer:
                            yield frame
                        elif len(buffer) < shuffle_buffer:
                            buffer.append(frame)
                        else:
                            index = rng.integers(len(buffer))
                            previous, buffer[index] = buffer[index], frame
                            yield previous
            rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()

    def iter_batches(self, batch_size, *, max_atoms, shuffle_buffer, seed):
        with closing(self.shuffled_frames(shuffle_buffer, seed)) as frames:
            while True:
                yield collate_peptides(
                    [next(frames) for _ in range(batch_size)], max_atoms
                )


def load_many_peptides(
    *,
    cache_dir,
    revision,
    topology_revision,
    forcefield_files,
    shards=None,
    sequences=None,
):
    forces_repo = "niklastr/many_peptides_forces"
    topology_repo = "transferable-samplers/many-peptides-md"
    api = HfApi()
    info = api.dataset_info(forces_repo, revision=revision)
    if shards is None:
        shards = sorted(
            entry.rfilename
            for entry in info.siblings
            if re.fullmatch(r"single_frames_forces/[0-9]{4}\.tar", entry.rfilename)
        )
    else:
        if isinstance(shards, str):
            shards = [shards]
        shards = [
            name
            if name.startswith("single_frames_forces/")
            else f"single_frames_forces/{name}"
            for name in shards
        ]
    topology_info = api.dataset_info(topology_repo, revision=topology_revision)
    download = partial(
        hf_hub_download,
        repo_type="dataset",
        cache_dir=str(Path(cache_dir) / "hub"),
    )
    pdb_tar = download(
        repo_id=topology_repo,
        revision=topology_info.sha,
        filename="pdb_tarfiles/train.tar",
    )
    dataset = ManyPeptidesDataset(
        shards,
        pdb_tar,
        sequences=sequences,
        forcefield_files=forcefield_files,
    )
    dataset.resolve_shard = lambda name: download(
        repo_id=forces_repo,
        revision=info.sha,
        filename=name,
    )
    dataset.metadata.update(
        repo_id=forces_repo,
        revision=info.sha,
        topology_repo_id=topology_repo,
        topology_revision=topology_info.sha,
        split="train",
        temperature_K=310.0,
        source_friction_per_ps=0.3,
    )
    return dataset
