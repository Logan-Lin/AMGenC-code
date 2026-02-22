from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from db import LogEntry, Run, save_run
from nn.charge import create_charge_module
from pipeline.analysis import (
    compute_step_charges,
    finalize_and_plot_charges,
    save_forward_trajectory,
)
from pipeline.flow_matching import FlowMatcher


def save_checkpoint(model: torch.nn.Module, run_id: str, epoch: int):
    ckpt_dir = os.path.join("checkpoints", run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{epoch:05d}.pt")
    torch.save({"epoch": epoch, "model": model.state_dict()}, path)
    return path


def load_checkpoint(model: torch.nn.Module, run_id: str, epoch: int):
    path = os.path.join("checkpoints", run_id, f"{epoch:05d}.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    return ckpt["epoch"]


def train(
    model: torch.nn.Module,
    flow_matcher: FlowMatcher,
    train_dataloader: DataLoader,
    run: Run,
    device: str = "cpu",
):
    cfg = run.trainer
    num_epochs = cfg.train_epoch
    lr = cfg.lr
    save_per_epoch = cfg.save_per_epoch
    analyze_trajectory = cfg.analyze_trajectory
    traj_n_steps = cfg.traj_n_steps
    analysis_temperature = getattr(cfg, 'analysis_temperature', 0.1)
    start_epoch = 0

    if cfg.use_checkpoint:
        ckpt_run = cfg.checkpoint_run_id if cfg.checkpoint_run_id else run.id
        start_epoch = load_checkpoint(model, ckpt_run, cfg.checkpoint_epoch)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    output_dir = os.path.join("outputs", run.id)
    os.makedirs(output_dir, exist_ok=True)

    if analyze_trajectory:
        # Set up charge module if available
        charge_mod = None
        if cfg.dataset.charge_module:
            elements = run.model.kwargs['elements']
            charge_mod = create_charge_module(cfg.dataset.charge_module, elements)
            charge_mod = charge_mod.to(device)

        timesteps_fwd = np.linspace(1, 0, traj_n_steps + 1)
        if charge_mod is not None:
            all_hard_charges = [[] for _ in range(traj_n_steps + 1)]
            all_soft_charges = [[] for _ in range(traj_n_steps + 1)]

        is_first_batch = True
        with torch.no_grad():
            for batch in train_dataloader:
                batch = batch.to(device)
                el_emb = flow_matcher.element_embedding.embed(batch.get_elements())
                batch = batch.update_attrs(element_emb=el_emb)

                # Save trajectory files for first batch only
                if is_first_batch:
                    save_forward_trajectory(batch, flow_matcher, traj_n_steps, output_dir)
                    is_first_batch = False

                # Charge analysis across all batches
                if charge_mod is not None:
                    source = flow_matcher.sample_source(batch)
                    clean_emb = batch.get_element_emb()
                    noise_emb = source.get_element_emb()
                    batch_size = batch.get_batch_size()
                    batch_indices = batch.get_batch_indices()
                    t_shape = [-1] + [1] * (clean_emb.dim() - 1)

                    for step_idx, t_val in enumerate(timesteps_fwd):
                        t_atom = torch.full(
                            (batch.get_num_atoms(),), t_val,
                            dtype=torch.float, device=device,
                        ).view(t_shape)
                        flow_el = flow_matcher.el_path.interpolate(noise_emb, clean_emb, t_atom)

                        hard, soft = compute_step_charges(
                            flow_el, batch_indices, batch_size,
                            charge_mod, analysis_temperature,
                        )
                        all_hard_charges[step_idx].append(hard.cpu())
                        all_soft_charges[step_idx].append(soft.cpu())

        if charge_mod is not None:
            analysis_dir = os.path.join(output_dir, "train_analysis")
            finalize_and_plot_charges(
                all_hard_charges, all_soft_charges, timesteps_fwd, analysis_dir,
                title_prefix="Forward Trajectory",
            )

    for epoch in range(start_epoch, num_epochs):
        epoch_start = datetime.now(timezone.utc)
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            batch = batch.to(device)

            # Embed elements
            el_emb = flow_matcher.element_embedding.embed(batch.get_elements())
            batch = batch.update_attrs(element_emb=el_emb)

            t = flow_matcher.sample_t(batch)
            flow_batch, (v_pos_target, v_el_target) = flow_matcher.compute_flow(batch, t)

            cond = batch.cond
            v_pos_pred, v_el_pred = model(flow_batch, t, cond=cond)

            # Per-sample MSE loss for positions
            batch_idx = flow_batch.get_batch_indices()
            diff_sq_pos = (v_pos_pred - v_pos_target).pow(2).sum(dim=-1)  # (N,)
            per_sample_loss = torch.zeros(flow_batch.get_batch_size(), device=device)
            for i in range(flow_batch.get_batch_size()):
                mask = batch_idx == i
                per_sample_loss[i] = diff_sq_pos[mask].mean()
            loss = per_sample_loss.mean()

            # Element velocity loss
            diff_sq_el = (v_el_pred - v_el_target).pow(2).sum(dim=-1)  # (N,)
            per_sample_el_loss = torch.zeros(flow_batch.get_batch_size(), device=device)
            for i in range(flow_batch.get_batch_size()):
                mask = batch_idx == i
                per_sample_el_loss[i] = diff_sq_el[mask].mean()
            loss = loss + per_sample_el_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_time = (datetime.now(timezone.utc) - epoch_start).total_seconds()
        run.logs.append(LogEntry(
            timestamp=datetime.now(timezone.utc), epoch=epoch + 1, loss=avg_loss,
            data={"epoch_time": epoch_time},
        ))
        save_run(run)

        print(f"  Epoch {epoch + 1}/{num_epochs}  loss={avg_loss:.6f}  time={epoch_time:.1f}s")

        if save_per_epoch and (epoch + 1) % save_per_epoch == 0:
            save_checkpoint(model, run.id, epoch + 1)

    save_checkpoint(model, run.id, num_epochs)
