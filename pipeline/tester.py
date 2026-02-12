from __future__ import annotations

import os
from datetime import datetime, timezone

import torch
from ase.io import write as ase_write
from torch.utils.data import DataLoader

from db import ResultEntry, Run
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
            trajectory = flow_matcher.generate(source, n_steps, model, cond=cond)

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

    infer_time = (datetime.now(timezone.utc) - infer_start).total_seconds()

    ase_write(os.path.join(output_dir, "generated.extxyz"), all_atoms)

    run.results.append(ResultEntry(
        timestamp=datetime.now(timezone.utc),
        metrics={"infer_time": infer_time},
        outputs={"num_samples": len(all_atoms)},
    ))
    run.save()
