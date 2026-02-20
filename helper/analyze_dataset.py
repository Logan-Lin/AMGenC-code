"""CLI tool for analyzing MaterialDataset statistics."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from ase.data import chemical_symbols
from ase.io import read as ase_read

from data import Sample


def load_samples(path: str) -> list[Sample]:
    """Load Sample objects from atoms.extxyz in the given directory."""
    atoms_file = str(Path(path) / "atoms.extxyz")
    frames = ase_read(atoms_file, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    return [Sample.from_ase_atoms(atoms) for atoms in frames]


def load_raw_properties(path: str, n_samples: int) -> list[dict]:
    """Load and merge all property JSON files from the dataset directory."""
    json_files = sorted(glob.glob(str(Path(path) / "*.json")))
    raw_props: list[dict] = [{} for _ in range(n_samples)]
    for jf in json_files:
        with open(jf) as fh:
            prop_list = json.load(fh)
        if len(prop_list) != n_samples:
            print(f"Warning: skipping {jf} ({len(prop_list)} entries, expected {n_samples})")
            continue
        for i, prop_dict in enumerate(prop_list):
            raw_props[i].update(prop_dict)
    return raw_props


def compute_element_distribution(samples: list[Sample]) -> dict[str, int]:
    """Count occurrences of each element across all samples, sorted descending."""
    counter: Counter[int] = Counter()
    for s in samples:
        counter.update(s.elements.tolist())
    return {
        chemical_symbols[z]: count
        for z, count in sorted(counter.items(), key=lambda x: x[1], reverse=True)
    }


def compute_density_stats(samples: list[Sample]) -> tuple[np.ndarray, float, float]:
    """Compute atomic number density (atoms/A^3) per sample. Returns (array, mean, std)."""
    densities = np.array([
        len(s.elements) / abs(torch.det(s.lattice).item())
        for s in samples
    ])
    return densities, float(densities.mean()), float(densities.std())


def compute_property_stats(raw_props: list[dict]) -> dict[str, dict]:
    """Compute offset (mean) and scale (std) for each scalar numerical property."""
    if not raw_props:
        return {}
    all_keys = list(raw_props[0].keys())
    result = {}
    for key in all_keys:
        values = []
        scalar = True
        for props in raw_props:
            v = props.get(key)
            if v is None or not isinstance(v, (int, float)):
                scalar = False
                break
        if not scalar:
            continue
        values = [float(props[key]) for props in raw_props]
        arr = np.array(values)
        result[key] = {
            "offset": float(arr.mean()),
            "scale": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return result


def build_report(path: str, samples: list[Sample], raw_props: list[dict]) -> str:
    """Build the full analysis report as a string."""
    lines: list[str] = []

    # Overview
    total_atoms = sum(len(s.elements) for s in samples)
    lines.append("=== Overview ===")
    lines.append(f"  Samples:     {len(samples)}")
    lines.append(f"  Total atoms: {total_atoms}")
    lines.append("")

    # Element distribution
    elem_dist = compute_element_distribution(samples)
    lines.append("=== Element Distribution ===")
    lines.append(f"  {'Element':<10} {'Count':>8} {'Frequency':>10}")
    lines.append(f"  {'-' * 10} {'-' * 8} {'-' * 10}")
    for sym, count in elem_dist.items():
        freq = count / total_atoms
        lines.append(f"  {sym:<10} {count:>8} {freq:>10.4f}")
    lines.append("")

    # Atomic number density
    densities, mean_d, std_d = compute_density_stats(samples)
    lines.append("=== Atomic Number Density ===")
    lines.append(f"  Mean: {mean_d:.6f} atoms/A^3")
    lines.append(f"  Std:  {std_d:.6f} atoms/A^3")
    lines.append(f"  Min:  {float(densities.min()):.6f} atoms/A^3")
    lines.append(f"  Max:  {float(densities.max()):.6f} atoms/A^3")
    lines.append("")

    # Property statistics
    prop_stats = compute_property_stats(raw_props)
    if prop_stats:
        lines.append("=== Property Statistics ===")
        name_w = max(len(n) for n in prop_stats) + 2
        lines.append(f"  {'Property':<{name_w}} {'Offset (mean)':>16} {'Scale (std)':>16} {'Min':>16} {'Max':>16}")
        lines.append(f"  {'-' * name_w} {'-' * 16} {'-' * 16} {'-' * 16} {'-' * 16}")
        for name, stats in prop_stats.items():
            lines.append(
                f"  {name:<{name_w}} {stats['offset']:>16.6f} {stats['scale']:>16.6f}"
                f" {stats['min']:>16.6f} {stats['max']:>16.6f}"
            )
    else:
        lines.append("No scalar numerical properties found.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze a MaterialDataset directory.")
    parser.add_argument("path", help="Path to the dataset directory (must contain atoms.extxyz)")
    args = parser.parse_args()

    path = args.path
    if not Path(path).is_dir():
        parser.error(f"Directory not found: {path}")
    if not (Path(path) / "atoms.extxyz").is_file():
        parser.error(f"No atoms.extxyz found in {path}")

    samples = load_samples(path)
    raw_props = load_raw_properties(path, len(samples))
    report = build_report(path, samples, raw_props)

    print(report)

    report_file = Path(path) / "report.txt"
    report_file.write_text(report)
    print(f"Report written to {report_file}")


if __name__ == "__main__":
    main()
