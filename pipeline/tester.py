from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import torch
from ase.io import write as ase_write
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

    # Set up charge analysis if enabled
    charge_mod = None
    if analyze_trajectory and cfg.dataset.charge_module:
        from nn.charge import create_charge_module
        elements = run.model.kwargs['elements']
        charge_mod = create_charge_module(cfg.dataset.charge_module, elements)
        charge_mod = charge_mod.to(device)
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
            trajectory, pred_cleans = flow_matcher.generate(source, n_steps, model, cond=cond)

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
            if charge_mod is not None:
                batch_size = pred_cleans[0].get_batch_size()
                for step_idx, pc in enumerate(pred_cleans):
                    pc_emb = pc.get_element_emb()
                    pc_bi = pc.get_batch_indices()

                    # Soft charges: apply softmax with low temperature to sharpen toward one-hot
                    soft_emb = torch.softmax(pc_emb / 0.1, dim=-1)
                    soft = charge_mod.batch_charge(soft_emb, pc_bi, batch_size)
                    all_soft_charges[step_idx].append(soft.cpu())

                    # Hard charges: argmax → one-hot → charge
                    hard_emb = torch.nn.functional.one_hot(
                        torch.argmax(pc_emb, dim=-1), pc_emb.shape[-1]
                    ).float()
                    hard = charge_mod.batch_charge(hard_emb, pc_bi, batch_size)
                    all_hard_charges[step_idx].append(hard.cpu())

    infer_time = (datetime.now(timezone.utc) - infer_start).total_seconds()

    ase_write(os.path.join(output_dir, "generated.extxyz"), all_atoms)

    # Plot charge analysis
    if charge_mod is not None:
        timesteps = np.linspace(0, 1, n_steps + 1)[1:]  # pred_clean at steps 1..n_steps
        hard_charges = torch.stack([torch.cat(step) for step in all_hard_charges]).numpy()  # (n_steps, N_samples)
        soft_charges = torch.stack([torch.cat(step) for step in all_soft_charges]).numpy()
        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        _plot_charge_convergence(timesteps, hard_charges, soft_charges, analysis_dir)
        _plot_charge_per_sample(timesteps, hard_charges, analysis_dir)
        _plot_charge_histogram(hard_charges[-1], analysis_dir)

    run.results.append(ResultEntry(
        timestamp=datetime.now(timezone.utc),
        metrics={"infer_time": infer_time},
        outputs={"num_samples": len(all_atoms)},
    ))
    save_run(run)


def _plot_charge_convergence(timesteps, hard_charges, soft_charges, output_dir):
    """Plot mean +/- std of estimated clean charge across samples at each step.

    Args:
        timesteps: (n_steps,) array of timestep values.
        hard_charges: (n_steps, N_samples) array of hard (argmax) charges.
        soft_charges: (n_steps, N_samples) array of soft (continuous) charges.
        output_dir: directory to save the figure.
    """
    import matplotlib.pyplot as plt

    hard_mean = hard_charges.mean(axis=1)
    hard_std = hard_charges.std(axis=1)
    soft_mean = soft_charges.mean(axis=1)
    soft_std = soft_charges.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='Ideal (0)')

    ax.plot(timesteps, hard_mean, color='#2563eb', label='Hard (argmax)')
    ax.fill_between(timesteps, hard_mean - hard_std, hard_mean + hard_std,
                     color='#2563eb', alpha=0.2)

    ax.plot(timesteps, soft_mean, color='#dc2626', label='Soft (continuous)')
    ax.fill_between(timesteps, soft_mean - soft_std, soft_mean + soft_std,
                     color='#dc2626', alpha=0.2)

    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Estimated Total Charge')
    ax.set_title('Predicted Clean Charge Convergence')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'charge_convergence.pdf'))
    plt.close(fig)


def _plot_charge_per_sample(timesteps, hard_charges, output_dir):
    """Plot individual sample charge trajectories.

    Args:
        timesteps: (n_steps,) array of timestep values.
        hard_charges: (n_steps, N_samples) array of hard charges.
        output_dir: directory to save the figure.
    """
    import matplotlib.pyplot as plt

    n_samples = hard_charges.shape[1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    for i in range(n_samples):
        ax.plot(timesteps, hard_charges[:, i], alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Estimated Total Charge')
    ax.set_title(f'Per-Sample Charge Trajectories (n={n_samples})')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'charge_per_sample.pdf'))
    plt.close(fig)


def _plot_charge_histogram(final_charges, output_dir):
    """Plot histogram of final sample charge values.

    Args:
        final_charges: (N_samples,) array of charge values at the last timestep.
        output_dir: directory to save the figure.
    """
    import matplotlib.pyplot as plt

    n_samples = len(final_charges)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    ax.hist(final_charges, bins=min(100, max(20, n_samples // 2)),
            color='#2563eb', alpha=0.7, edgecolor='white', linewidth=0.5)

    mean = final_charges.mean()
    std = final_charges.std()
    ax.axvline(mean, color='#dc2626', linewidth=1.5, label=f'Mean: {mean:.3f}')
    ax.set_xlabel('Total Charge')
    ax.set_ylabel('Count')
    ax.set_title(f'Final Sample Charge Distribution (n={n_samples}, std={std:.3f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'charge_histogram.pdf'))
    plt.close(fig)
