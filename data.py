from __future__ import annotations

import dataclasses
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from ase import Atoms as AseAtoms
from ase.io import read as ase_read
from torch.utils.data import DataLoader, Dataset

from utils.neighborlist import Neighborlist, _positions_into_cell


@dataclass
class Sample:
    """A single material sample."""

    elements: torch.LongTensor  # (N,) atomic numbers
    positions: torch.FloatTensor  # (N, 3) Cartesian coordinates
    lattice: torch.FloatTensor  # (3, 3) unit cell vectors as rows
    pbc: tuple[bool, bool, bool] = (True, True, True)

    neighborlist: Optional[Neighborlist] = field(default=None, repr=False)
    element_emb: Optional[torch.FloatTensor] = field(default=None, repr=False)
    properties: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_ase_atoms(atoms: AseAtoms, properties: dict[str, float] | None = None) -> Sample:
        return Sample(
            elements=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
            positions=torch.tensor(atoms.get_positions(), dtype=torch.float),
            lattice=torch.tensor(np.array(atoms.get_cell(complete=True)), dtype=torch.float),
            pbc=tuple(atoms.get_pbc().tolist()),
            properties=properties or {},
        )

    def to_ase_atoms(self) -> AseAtoms:
        return AseAtoms(
            numbers=self.elements.detach().cpu().numpy(),
            positions=self.positions.detach().cpu().numpy(),
            cell=self.lattice.detach().cpu().numpy(),
            pbc=self.pbc,
        )

    def to(self, device) -> Sample:
        def _mv(t):
            return t.to(device) if t is not None else None

        nl = self.neighborlist.to(device) if self.neighborlist is not None else None
        return dataclasses.replace(
            self,
            elements=self.elements.to(device),
            positions=self.positions.to(device),
            lattice=self.lattice.to(device),
            neighborlist=nl,
            element_emb=_mv(self.element_emb),
        )

    def update_attrs(self, **kwargs) -> Sample:
        return dataclasses.replace(self, **kwargs)

    def get_positions(self) -> torch.FloatTensor:
        return self.positions

    def get_elements(self) -> torch.LongTensor:
        return self.elements

    def get_element_emb(self) -> Optional[torch.FloatTensor]:
        return self.element_emb

    def get_num_atoms(self) -> int:
        return len(self.elements)

    def get_batch_size(self) -> int:
        return 1

    def get_batch_indices(self) -> torch.LongTensor:
        return torch.zeros(self.get_num_atoms(), dtype=torch.long, device=self.positions.device)

    def get_edges(self, r_cut: float) -> tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        if self.neighborlist is None:
            self.neighborlist = Neighborlist(self.lattice, self.pbc)
        return self.neighborlist.get_edges(self.positions, r_cut)

    def randomize_uniform(self) -> Sample:
        x = torch.rand_like(self.positions) @ self.lattice
        return self.update_attrs(positions=x)

    def back_to_cell(self) -> Sample:
        return self.update_attrs(positions=_positions_into_cell(self.positions, self.lattice))

    def remove_mean(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x - torch.mean(x, dim=0, keepdim=True)

    def cal_velocity(self, target_positions: torch.FloatTensor) -> torch.FloatTensor:
        """PBC-aware velocity from self.positions to target_positions."""
        delta = target_positions - self.positions
        frac_delta = torch.linalg.solve(self.lattice.T, delta.T).T
        frac_delta = frac_delta - torch.round(frac_delta)
        return frac_delta @ self.lattice


class Batch:
    """A merged batch of Samples."""

    def __init__(self, samples: list[Sample]):
        self.samples = samples
        self._build_cache()

    def _build_cache(self):
        samples = self.samples
        elements_list, positions_list, lattice_list = [], [], []
        batch_indices_list = []
        num_atoms_per_sample = []
        element_emb_list = []
        has_emb = all(s.element_emb is not None for s in samples)

        for i, s in enumerate(samples):
            n = s.get_num_atoms()
            elements_list.append(s.elements)
            positions_list.append(s.positions)
            lattice_list.append(s.lattice)
            batch_indices_list.append(torch.full((n,), i, dtype=torch.long, device=s.positions.device))
            num_atoms_per_sample.append(n)
            if has_emb:
                element_emb_list.append(s.element_emb)

        self.elements = torch.cat(elements_list)
        self.positions = torch.cat(positions_list)
        self.lattice = torch.stack(lattice_list)
        self.pbc = samples[0].pbc
        self.batch_indices = torch.cat(batch_indices_list)
        self.num_atoms_per_sample = num_atoms_per_sample
        self.element_emb = torch.cat(element_emb_list) if has_emb else None

    def update_attrs(self, **kwargs) -> Batch:
        per_atom_keys = {"positions", "elements", "element_emb"}
        sample_kwargs_list = [{} for _ in self.samples]

        for key, value in kwargs.items():
            if key in per_atom_keys and value is not None:
                offset = 0
                for i, n in enumerate(self.num_atoms_per_sample):
                    sample_kwargs_list[i][key] = value[offset:offset + n]
                    offset += n
            else:
                for i in range(len(self.samples)):
                    sample_kwargs_list[i][key] = value

        new_samples = [
            s.update_attrs(**kw) if kw else s
            for s, kw in zip(self.samples, sample_kwargs_list)
        ]
        return Batch(new_samples)

    def to(self, device) -> Batch:
        return Batch([s.to(device) for s in self.samples])

    def to_samples(self) -> list[Sample]:
        return list(self.samples)

    def get_positions(self) -> torch.FloatTensor:
        return self.positions

    def get_elements(self) -> torch.LongTensor:
        return self.elements

    def get_element_emb(self) -> Optional[torch.FloatTensor]:
        return self.element_emb

    def get_batch_size(self) -> int:
        return len(self.samples)

    def get_num_atoms(self) -> int:
        return len(self.elements)

    def get_batch_indices(self) -> torch.LongTensor:
        return self.batch_indices

    @torch.no_grad()
    def get_edges(self, r_cut: float) -> tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        edge_src_list, edge_dst_list, edge_off_list = [], [], []
        offset = 0
        for s in self.samples:
            src, dst, offs = s.get_edges(r_cut)
            edge_src_list.append(src + offset)
            edge_dst_list.append(dst + offset)
            edge_off_list.append(offs)
            offset += s.get_num_atoms()
        return (
            torch.cat(edge_src_list),
            torch.cat(edge_dst_list),
            torch.cat(edge_off_list),
        )

    def randomize_uniform(self) -> Batch:
        return Batch([s.randomize_uniform() for s in self.samples])

    def back_to_cell(self) -> Batch:
        return Batch([s.back_to_cell() for s in self.samples])

    def remove_mean(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean = torch.zeros_like(x)
        for i in range(self.get_batch_size()):
            mask = self.batch_indices == i
            mean[mask] = torch.mean(x[mask], dim=0, keepdim=True)
        return x - mean

    def cal_velocity(self, target_positions: torch.FloatTensor) -> torch.FloatTensor:
        """PBC-aware velocity, delegating to per-sample cal_velocity."""
        velocities = []
        offset = 0
        for s in self.samples:
            n = s.get_num_atoms()
            v = s.cal_velocity(target_positions[offset:offset + n])
            velocities.append(v)
            offset += n
        return torch.cat(velocities, dim=0)

    def get_condition_tensor(
        self,
        property_names: list[str],
        property_stats: dict[str, tuple[float, float]],
    ) -> torch.FloatTensor:
        """Build (B, d_cond) normalized condition tensor from sample properties.

        Args:
            property_names: ordered list of property keys to include.
            property_stats: {name: (offset, scale)} for normalization: (value - offset) / scale.

        Returns:
            (batch_size, len(property_names)) float tensor on same device as positions.
        """
        device = self.positions.device
        B = self.get_batch_size()
        d = len(property_names)
        cond = torch.zeros(B, d, device=device)
        for i, s in enumerate(self.samples):
            for j, name in enumerate(property_names):
                val = s.properties.get(name, 0.0)
                offset, scale = property_stats.get(name, (0.0, 1.0))
                cond[i, j] = (val - offset) / scale
        return cond


class MaterialDataset(Dataset):
    """Dataset of material samples loaded from extxyz files with optional property JSONs."""

    def __init__(self, path: str):
        super().__init__()
        files = sorted(glob.glob(os.path.join(path, "*.extxyz")))
        if len(files) == 0:
            raise FileNotFoundError(f"No .extxyz files found in {path}")

        # Load all frames from all extxyz files
        samples: list[Sample] = []
        for f in files:
            frames = ase_read(f, index=":")
            if not isinstance(frames, list):
                frames = [frames]
            for atoms in frames:
                samples.append(Sample.from_ase_atoms(atoms))

        # Load and merge property JSON files
        json_files = sorted(glob.glob(os.path.join(path, "*.json")))
        for jf in json_files:
            with open(jf) as fh:
                prop_list = json.load(fh)
            if len(prop_list) != len(samples):
                raise ValueError(
                    f"Property file {jf} has {len(prop_list)} entries but dataset has {len(samples)} samples"
                )
            for i, prop_dict in enumerate(prop_list):
                samples[i].properties.update(prop_dict)

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


class MaterialCollateFn:
    """Collate function for DataLoader."""

    def __init__(self, device: str = "cpu", init_r_cut: float | None = None):
        self.device = device
        self.init_r_cut = init_r_cut

    def __call__(self, sample_batch: list[Sample]) -> Batch:
        if self.init_r_cut is not None:
            for s in sample_batch:
                if s.neighborlist is None:
                    s.neighborlist = Neighborlist(s.lattice, s.pbc, init_r_cut=self.init_r_cut)
                elif s.neighborlist.init_r_cut != self.init_r_cut:
                    s.neighborlist.set_init_r_cut(self.init_r_cut)
        return Batch(sample_batch).to(self.device)


def create_dataloader(dataset_cfg, device: str, shuffle: bool = False) -> DataLoader:
    """Create DataLoader from Dataset configuration document.

    Args:
        dataset_cfg: Dataset embedded document with path, batch_size, init_r_cut.
        device: Device string for collate function.
        shuffle: Whether to shuffle data.

    Returns:
        Configured DataLoader instance.
    """
    dataset = MaterialDataset(path=dataset_cfg.path)
    collate_fn = MaterialCollateFn(device=device, init_r_cut=dataset_cfg.init_r_cut)
    return DataLoader(dataset, batch_size=dataset_cfg.batch_size, shuffle=shuffle, collate_fn=collate_fn)


if __name__ == "__main__":
    ds = MaterialDataset("data/bmp-sample")
    print(f"Dataset size: {len(ds)}")
    s = ds[0]
    print(f"elements={s.elements.shape}, positions={s.positions.shape}, properties={s.properties}")
