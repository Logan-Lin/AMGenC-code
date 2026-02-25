from __future__ import annotations

import os
from datetime import datetime, timezone

import torch

from db import ResultEntry, Run, save_run
from nn.charge import create_charge_module
from pipeline.analysis import (
    plot_charge_convergence,
    plot_charge_histogram,
    plot_charge_per_sample,
)


def analyze(run: Run):
    """Run charge analysis on saved trajectory data.

    Loads convergence.pt (all-sample hard charges) and per-sample element
    logit files from the source run's infer_traj directory.
    """
    cfg = run.analyzer
    temperature = cfg.analysis_temperature

    # Determine source run
    source_run_id = cfg.source_run_id or run.id
    if source_run_id != run.id:
        source_run = Run.objects(id=source_run_id).first()
        if source_run is None:
            raise ValueError(f"Source run {source_run_id!r} not found")
    else:
        source_run = run

    if source_run.tester is None:
        raise ValueError(f"Source run {source_run_id} has no tester configuration")
    if not source_run.tester.save_trajectory:
        raise ValueError(f"Source run {source_run_id} did not save trajectory data")

    traj_dir = os.path.join("outputs", source_run_id, "infer_traj")
    if not os.path.isdir(traj_dir):
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

    # Load convergence data (all-sample hard charges)
    convergence_path = os.path.join(traj_dir, "convergence.pt")
    if not os.path.isfile(convergence_path):
        raise FileNotFoundError(f"Convergence data not found: {convergence_path}")
    convergence = torch.load(convergence_path, map_location="cpu", weights_only=True)
    hard_charges_all = convergence["hard_charges"].numpy()  # (n_steps, n_total)
    timesteps = convergence["timesteps"].numpy()  # (n_steps,)

    n_steps = hard_charges_all.shape[0]

    # Create charge module from source run's tester dataset config
    charge_module_name = source_run.tester.dataset.charge_module
    if not charge_module_name:
        raise ValueError(f"Source run {source_run_id} tester dataset has no charge_module")
    elements = source_run.model.kwargs["elements"]
    charge_mod = create_charge_module(charge_module_name, elements)

    # Load per-sample logit files for per-sample plot
    sample_files = sorted(
        f for f in os.listdir(traj_dir)
        if f.endswith(".pt") and f != "convergence.pt"
    )

    # Compute per-sample hard charges from individual logits
    per_sample_hard = [[] for _ in range(n_steps)]
    for fname in sample_files:
        data = torch.load(
            os.path.join(traj_dir, fname), map_location="cpu", weights_only=True,
        )
        logit_list = data["element_logits"]
        for step_idx, logits in enumerate(logit_list):
            hard_emb = torch.nn.functional.one_hot(
                torch.argmax(logits, dim=-1), logits.shape[-1],
            ).float()
            hard = charge_mod.per_atom_charge(hard_emb).sum()
            per_sample_hard[step_idx].append(hard)

    # Produce plots
    output_dir = os.path.join("outputs", run.id, "infer_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Convergence and histogram use all-sample hard charges
    # Pass hard_charges_all as both hard and soft (soft curve will duplicate hard)
    plot_charge_convergence(
        timesteps, hard_charges_all, output_dir,
        title="Predicted Clean Charge Convergence",
    )
    plot_charge_histogram(
        hard_charges_all[-1], output_dir,
        title="Final Charge Distribution",
    )

    # Per-sample plot uses the subset with individual logits
    if sample_files:
        per_sample_hard_np = torch.stack(
            [torch.stack(step) for step in per_sample_hard]
        ).numpy()
        plot_charge_per_sample(
            timesteps, per_sample_hard_np, output_dir,
            title="Per-Sample Charge Trajectories",
        )

    # Record analysis result
    n_total = hard_charges_all.shape[1]
    run.results.append(ResultEntry(
        timestamp=datetime.now(timezone.utc),
        metrics={
            "source_run_id": source_run_id,
            "analysis_temperature": temperature,
            "n_total_samples": n_total,
            "n_individual_samples": len(sample_files),
        },
        outputs={},
    ))
    save_run(run)
