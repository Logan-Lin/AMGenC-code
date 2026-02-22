from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import torch
from ase.io import write as ase_write
from torch.utils.data import DataLoader

from db import ResultEntry, Run, save_run
from pipeline.analysis import compute_step_charges, finalize_and_plot_charges
from pipeline.flow_matching import FlowMatcher
from pipeline.trainer import load_checkpoint


def test(
    model: torch.nn.Module,
    flow_matcher: FlowMatcher,
    test_dataloader: DataLoader,
    run: Run,
    device: str = "cpu",
):
    cfg = run.tester
    n_steps = cfg.n_steps
    save_trajectory = cfg.save_trajectory
    analyze_trajectory = cfg.analyze_trajectory

    if cfg.use_checkpoint:
        ckpt_run = cfg.checkpoint_run_id if cfg.checkpoint_run_id else run.id
        load_checkpoint(model, ckpt_run, cfg.checkpoint_epoch)

    model = model.to(device)
    model.eval()

    output_dir = os.path.join("outputs", run.id)
    os.makedirs(output_dir, exist_ok=True)
    if save_trajectory:
        traj_dir = os.path.join(output_dir, "infer_traj")
        os.makedirs(traj_dir, exist_ok=True)

    # Set up charge module if needed (for analysis and/or PCFM projection)
    use_pcfm = getattr(cfg, 'use_pcfm', False)
    pcfm_temperature = getattr(cfg, 'pcfm_temperature', 0.1)
    analysis_temperature = getattr(cfg, 'analysis_temperature', 0.1)
    needs_charge_mod = (analyze_trajectory and cfg.dataset.charge_module) or use_pcfm

    charge_mod = None
    if needs_charge_mod:
        if not cfg.dataset.charge_module:
            raise ValueError("use_pcfm requires a charge_module to be set in the dataset config")
        from nn.charge import create_charge_module
        elements = run.model.kwargs['elements']
        charge_mod = create_charge_module(cfg.dataset.charge_module, elements)
        charge_mod = charge_mod.to(device)

    if analyze_trajectory and charge_mod is not None:
        # Per-step lists accumulating per-sample charges across all batches
        all_hard_charges = [[] for _ in range(n_steps)]
        all_soft_charges = [[] for _ in range(n_steps)]

    all_atoms = []
    is_first_batch = True
    infer_start = datetime.now(timezone.utc)

    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)

            # Embed elements
            el_emb = flow_matcher.element_embedding.embed(batch.get_elements())
            batch = batch.update_attrs(element_emb=el_emb)

            cond = batch.cond
            source = flow_matcher.sample_source(batch)
            pcfm_charge_mod = charge_mod if use_pcfm else None
            trajectory, pred_cleans = flow_matcher.generate(
                source, n_steps, model, cond=cond,
                charge_module=pcfm_charge_mod, pcfm_temperature=pcfm_temperature,
            )

            final = trajectory[-1]
            for s in final.to_samples():
                all_atoms.append(s.back_to_cell().to_ase_atoms())

            if save_trajectory and is_first_batch:
                for i, _ in enumerate(final.to_samples()):
                    traj_atoms = []
                    for step_batch in trajectory:
                        traj_atoms.append(step_batch.to_samples()[i].back_to_cell().to_ase_atoms())
                    ase_write(os.path.join(traj_dir, f"{i:05d}.extxyz"), traj_atoms)
                is_first_batch = False

            # Accumulate charge analysis across all batches
            if analyze_trajectory and charge_mod is not None:
                batch_size = pred_cleans[0].get_batch_size()
                for step_idx, pc in enumerate(pred_cleans):
                    hard, soft = compute_step_charges(
                        pc.get_element_emb(), pc.get_batch_indices(),
                        batch_size, charge_mod, analysis_temperature,
                    )
                    all_hard_charges[step_idx].append(hard.cpu())
                    all_soft_charges[step_idx].append(soft.cpu())

    infer_time = (datetime.now(timezone.utc) - infer_start).total_seconds()

    ase_write(os.path.join(output_dir, "generated.extxyz"), all_atoms)

    # Plot charge analysis
    if analyze_trajectory and charge_mod is not None:
        timesteps = np.linspace(0, 1, n_steps + 1)[1:]  # pred_clean at steps 1..n_steps
        analysis_dir = os.path.join(output_dir, "infer_analysis")
        finalize_and_plot_charges(all_hard_charges, all_soft_charges, timesteps, analysis_dir,
                                  title_prefix="Predicted Clean")

    run.results.append(ResultEntry(
        timestamp=datetime.now(timezone.utc),
        metrics={"infer_time": infer_time},
        outputs={"num_samples": len(all_atoms)},
    ))
    save_run(run)
