from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import torch
from ase.io import write as ase_write
from ase.io.formats import string2index
from torch.utils.data import DataLoader

from db import ResultEntry, Run, save_run
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

    # Set up charge module for PCFM projection and/or trajectory saving
    use_pcfm = getattr(cfg, 'use_pcfm', False)
    pcfm_temperature = getattr(cfg, 'pcfm_temperature', 0.1)
    needs_charge_mod = use_pcfm or save_trajectory

    charge_mod = None
    if needs_charge_mod:
        if not cfg.dataset.charge_module:
            raise ValueError("save_trajectory and use_pcfm require a charge_module in the dataset config")
        from nn.charge import create_charge_module
        elements = run.model.kwargs['elements']
        charge_mod = create_charge_module(cfg.dataset.charge_module, elements)
        charge_mod = charge_mod.to(device)

    # Trajectory data saving setup
    if save_trajectory:
        traj_dir = os.path.join(output_dir, "infer_traj")
        os.makedirs(traj_dir, exist_ok=True)
        save_index_str = cfg.save_index or ":"
        save_index = string2index(save_index_str)
        all_sample_logits: list[list[torch.Tensor]] = []
        all_traj_atoms: list[list] = []  # per-sample list of ASE Atoms across steps
        # Per-step hard charges across all batches
        all_hard_charges: list[list[torch.Tensor]] = [[] for _ in range(n_steps)]

    all_atoms = []
    infer_time = 0.0

    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            batch_start = datetime.now(timezone.utc)

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

            infer_time += (datetime.now(timezone.utc) - batch_start).total_seconds()

            if save_trajectory:
                batch_size = pred_cleans[0].get_batch_size()

                # Compute hard charges for ALL samples at each step
                for step_idx, pc in enumerate(pred_cleans):
                    el = pc.get_element_emb()
                    hard_emb = torch.nn.functional.one_hot(
                        torch.argmax(el, dim=-1), el.shape[-1],
                    ).float()
                    hard = charge_mod.batch_charge(
                        hard_emb, pc.get_batch_indices(), batch_size,
                    )
                    all_hard_charges[step_idx].append(hard.cpu())

                # Accumulate raw element logits per sample (for subset saving)
                for local_idx in range(batch_size):
                    logits_per_step = []
                    for pc in pred_cleans:
                        mask = pc.get_batch_indices() == local_idx
                        logits_per_step.append(pc.get_element_emb()[mask].cpu())
                    all_sample_logits.append(logits_per_step)

                # Accumulate per-sample ASE trajectory atoms
                samples_per_step = [step.to_samples() for step in trajectory]
                for local_idx in range(batch_size):
                    atoms_list = [
                        samples_per_step[step_idx][local_idx].back_to_cell().to_ase_atoms()
                        for step_idx in range(len(trajectory))
                    ]
                    all_traj_atoms.append(atoms_list)

    ase_write(os.path.join(output_dir, "generated.extxyz"), all_atoms)

    if save_trajectory:
        # Save convergence data (all samples)
        hard_charges = torch.stack([torch.cat(step) for step in all_hard_charges])
        timesteps = torch.from_numpy(
            np.linspace(0, 1, n_steps + 1)[1:].astype(np.float32)
        )
        torch.save(
            {"hard_charges": hard_charges, "timesteps": timesteps},
            os.path.join(traj_dir, "convergence.pt"),
        )

        # Save per-sample element logit and trajectory files for the selected subset
        selected_logits = all_sample_logits[save_index]
        if not isinstance(selected_logits, list):
            selected_logits = [selected_logits]
        selected_traj = all_traj_atoms[save_index]
        if not isinstance(selected_traj, list):
            selected_traj = [selected_traj]
        for file_idx, (logit_list, traj_atoms) in enumerate(zip(selected_logits, selected_traj)):
            torch.save(
                {"element_logits": logit_list},
                os.path.join(traj_dir, f"{file_idx:05d}.pt"),
            )
            ase_write(os.path.join(traj_dir, f"{file_idx:05d}.extxyz"), traj_atoms)

    run.results.append(ResultEntry(
        timestamp=datetime.now(timezone.utc),
        metrics={"infer_time": infer_time},
        outputs={"num_samples": len(all_atoms)},
    ))
    save_run(run)
