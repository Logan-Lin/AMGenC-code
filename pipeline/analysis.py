from __future__ import annotations

import os

import numpy as np
import torch
from ase.io import write as ase_write

from pipeline.flow_matching import FlowMatcher


def save_forward_trajectory(
    batch,
    flow_matcher: FlowMatcher,
    n_steps: int,
    output_dir: str,
):
    """Save forward (noising) trajectory for a batch of samples."""
    traj_dir = os.path.join(output_dir, "train_analysis", "traj")
    os.makedirs(traj_dir, exist_ok=True)

    device = batch.get_positions().device
    source = flow_matcher.sample_source(batch)
    clean_positions = batch.get_positions()

    batch_indices = batch.get_batch_indices()
    t_shape = [-1] + [1] * (len(clean_positions.shape) - 1)

    timesteps = np.linspace(1, 0, n_steps + 1)
    trajectory = []

    for t_val in timesteps:
        t_atom = torch.full((batch.get_num_atoms(),), t_val, dtype=torch.float, device=device).view(t_shape)
        flow_positions = flow_matcher.pos_path.interpolate(source, clean_positions, t_atom)

        clean_emb = batch.get_element_emb()
        noise_emb = source.get_element_emb()
        flow_el = flow_matcher.el_path.interpolate(noise_emb, clean_emb, t_atom)

        trajectory.append(batch.update_attrs(
            positions=flow_positions,
            element_emb=flow_el,
            elements=flow_matcher.element_embedding.unembed(flow_el),
        ))

    for i, _ in enumerate(batch.to_samples()):
        traj_atoms = []
        for step_batch in trajectory:
            traj_atoms.append(step_batch.to_samples()[i].back_to_cell().to_ase_atoms())
        ase_write(os.path.join(traj_dir, f"{i:05d}.extxyz"), traj_atoms)


def compute_step_charges(
    element_emb: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
    charge_mod,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute hard and soft per-sample total charges for one step.

    Returns:
        (hard_charges, soft_charges) each of shape (batch_size,)
    """
    # Soft charges: apply softmax with configurable temperature
    soft_emb = torch.softmax(element_emb / temperature, dim=-1)
    soft = charge_mod.batch_charge(soft_emb, batch_indices, batch_size)

    # Hard charges: argmax -> one-hot -> charge
    hard_emb = torch.nn.functional.one_hot(
        torch.argmax(element_emb, dim=-1), element_emb.shape[-1]
    ).float()
    hard = charge_mod.batch_charge(hard_emb, batch_indices, batch_size)

    return hard, soft


def finalize_and_plot_charges(
    all_hard_charges: list[list[torch.Tensor]],
    all_soft_charges: list[list[torch.Tensor]],
    timesteps: np.ndarray,
    output_dir: str,
    title_prefix: str = "",
):
    """Stack accumulated charges and produce all analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    hard_charges = torch.stack([torch.cat(step) for step in all_hard_charges]).numpy()
    soft_charges = torch.stack([torch.cat(step) for step in all_soft_charges]).numpy()
    pfx = f"{title_prefix} " if title_prefix else ""
    plot_charge_convergence(timesteps, hard_charges, soft_charges, output_dir,
                            title=f"{pfx}Charge Convergence")
    plot_charge_per_sample(timesteps, hard_charges, output_dir,
                           title=f"{pfx}Per-Sample Charge Trajectories")
    plot_charge_histogram(hard_charges[-1], output_dir,
                          title=f"{pfx}Final Charge Distribution")


def plot_charge_convergence(timesteps, hard_charges, soft_charges, output_dir,
                            title="Charge Convergence"):
    """Plot mean +/- std of estimated clean charge across samples at each step.

    Args:
        timesteps: (n_steps,) array of timestep values.
        hard_charges: (n_steps, N_samples) array of hard (argmax) charges.
        soft_charges: (n_steps, N_samples) array of soft (continuous) charges.
        output_dir: directory to save the figure.
        title: plot title.
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
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'charge_convergence.pdf'))
    plt.close(fig)


def plot_charge_per_sample(timesteps, hard_charges, output_dir,
                           title="Per-Sample Charge Trajectories"):
    """Plot individual sample charge trajectories.

    Args:
        timesteps: (n_steps,) array of timestep values.
        hard_charges: (n_steps, N_samples) array of hard charges.
        output_dir: directory to save the figure.
        title: plot title (n_samples will be appended).
    """
    import matplotlib.pyplot as plt

    n_samples = hard_charges.shape[1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    for i in range(n_samples):
        ax.plot(timesteps, hard_charges[:, i], alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Estimated Total Charge')
    ax.set_title(f'{title} (n={n_samples})')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'charge_per_sample.pdf'))
    plt.close(fig)


def plot_charge_histogram(final_charges, output_dir,
                          title="Final Charge Distribution"):
    """Plot histogram of final sample charge values.

    Args:
        final_charges: (N_samples,) array of charge values at the last timestep.
        output_dir: directory to save the figure.
        title: plot title (n_samples and std will be appended).
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
    ax.set_title(f'{title} (n={n_samples}, std={std:.3f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'charge_histogram.pdf'))
    plt.close(fig)
