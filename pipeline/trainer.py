from __future__ import annotations

import os
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader

from db import LogEntry, Run
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
    start_epoch = 0

    if cfg.use_checkpoint:
        ckpt_run = cfg.checkpoint_run_id if cfg.checkpoint_run_id else run.id
        start_epoch = load_checkpoint(model, ckpt_run, cfg.checkpoint_epoch)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Gather property names and stats from dataset config
    property_names = []
    property_stats = {}
    if cfg.dataset.properties:
        for p in cfg.dataset.properties:
            property_names.append(p.name)
            property_stats[p.name] = (p.offset, p.scale)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = datetime.now(timezone.utc)
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            batch = batch.to(device)

            # Embed elements
            if flow_matcher.element_embedding is not None:
                el_emb = flow_matcher.element_embedding.embed(batch.get_elements())
                batch = batch.update_attrs(element_emb=el_emb)

            t = flow_matcher.sample_t(batch)
            flow_batch, (v_pos_target, v_el_target) = flow_matcher.compute_flow(batch, t)

            # Condition tensor
            cond = None
            if property_names:
                cond = batch.get_condition_tensor(property_names, property_stats)

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
            if v_el_target is not None and v_el_pred is not None:
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
        run.save()

        print(f"  Epoch {epoch + 1}/{num_epochs}  loss={avg_loss:.6f}  time={epoch_time:.1f}s")

        if save_per_epoch and (epoch + 1) % save_per_epoch == 0:
            save_checkpoint(model, run.id, epoch + 1)

    save_checkpoint(model, run.id, num_epochs)
