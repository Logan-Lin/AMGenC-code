from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from ase.io import write as ase_write

from db import ResultEntry, Run, save_run
from nn.charge import ChargeModule, create_charge_module
from pipeline.flow_matching import FlowMatcher


# ======================================================================
# Standalone helpers (used by trainer and other modules)
# ======================================================================

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
    soft_emb = torch.softmax(element_emb / temperature, dim=-1)
    soft = charge_mod.batch_charge(soft_emb, batch_indices, batch_size)

    hard_emb = F.one_hot(
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
    plot_charge_convergence(timesteps, hard_charges, output_dir,
                            soft_charges=soft_charges, title=f"{pfx}Charge Convergence")
    plot_charge_per_sample(timesteps, hard_charges, output_dir,
                           title=f"{pfx}Per-Sample Charge Trajectories")
    plot_charge_histogram(hard_charges[-1], output_dir,
                          title=f"{pfx}Final Charge Distribution")


# ======================================================================
# Plot functions
# ======================================================================

def plot_charge_convergence(timesteps, hard_charges, output_dir,
                            soft_charges=None, title="Charge Convergence"):
    """Plot mean +/- std of estimated clean charge across samples at each step.

    Args:
        timesteps: (n_steps,) array of timestep values.
        hard_charges: (n_steps, N_samples) array of hard (argmax) charges.
        output_dir: directory to save the figure.
        soft_charges: optional (n_steps, N_samples) array of soft (continuous) charges.
        title: plot title.
    """
    import matplotlib.pyplot as plt

    hard_mean = hard_charges.mean(axis=1)
    hard_std = hard_charges.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='Ideal (0)')

    ax.plot(timesteps, hard_mean, color='#2563eb', label='Hard (argmax)')
    ax.fill_between(timesteps, hard_mean - hard_std, hard_mean + hard_std,
                     color='#2563eb', alpha=0.2)

    if soft_charges is not None:
        soft_mean = soft_charges.mean(axis=1)
        soft_std = soft_charges.std(axis=1)
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


# ======================================================================
# TrajectoryAnalyzer
# ======================================================================

class TrajectoryAnalyzer:
    """Post-hoc analysis of saved inference trajectories.

    Loads convergence data and per-sample element logit trajectories from a
    tester run, computes charge and stability diagnostics, and produces plots.
    """

    def __init__(self, run: Run):
        self.run = run
        cfg = run.analyzer
        self.temperature = cfg.analysis_temperature

        # Resolve source run
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

        self.traj_dir = os.path.join("outputs", source_run_id, "infer_traj")
        if not os.path.isdir(self.traj_dir):
            raise FileNotFoundError(f"Trajectory directory not found: {self.traj_dir}")

        self.output_dir = os.path.join("outputs", run.id, "infer_analysis")
        os.makedirs(self.output_dir, exist_ok=True)

        # Load convergence data (all-sample hard charges from tester)
        convergence_path = os.path.join(self.traj_dir, "convergence.pt")
        if not os.path.isfile(convergence_path):
            raise FileNotFoundError(f"Convergence data not found: {convergence_path}")
        convergence = torch.load(convergence_path, map_location="cpu", weights_only=True)
        self.hard_charges_all = convergence["hard_charges"].numpy()  # (n_steps, n_total)
        self.timesteps = convergence["timesteps"].numpy()  # (n_steps,)
        self.n_steps = self.hard_charges_all.shape[0]

        # Create charge module
        charge_module_name = source_run.tester.dataset.charge_module
        if not charge_module_name:
            raise ValueError(f"Source run {source_run_id} tester dataset has no charge_module")
        elements = source_run.model.kwargs["elements"]
        self.charge_mod = create_charge_module(charge_module_name, elements)

        # Load per-sample logit files
        sample_files = sorted(
            f for f in os.listdir(self.traj_dir)
            if f.endswith(".pt") and f != "convergence.pt"
        )
        self.sample_logits: list[list[torch.Tensor]] = []
        for fname in sample_files:
            data = torch.load(
                os.path.join(self.traj_dir, fname), map_location="cpu", weights_only=True,
            )
            self.sample_logits.append(data["element_logits"])
        self.n_samples = len(self.sample_logits)

        # Diagnostic results (populated by _analyze_* methods)
        self.soft_charges: np.ndarray | None = None      # (n_samples, n_steps)
        self.hard_charges_ps: np.ndarray | None = None    # (n_samples, n_steps)
        self.entropy_mean: np.ndarray | None = None       # (n_samples, n_steps)
        self.entropy_min: np.ndarray | None = None        # (n_samples, n_steps)
        self.confidence_mean: np.ndarray | None = None    # (n_samples, n_steps)
        self.confidence_min: np.ndarray | None = None     # (n_samples, n_steps)
        self.switch_fractions: np.ndarray | None = None   # (n_samples, n_steps-1)
        self.pcfm_scales: np.ndarray | None = None        # (n_samples, n_steps)
        self.grad_norms: np.ndarray | None = None         # (n_samples, n_steps)
        self.correction_norms: np.ndarray | None = None   # (n_samples, n_steps)

    def main(self):
        """Run all analyses and save results."""
        self._analyze_charge_convergence()
        if self.n_samples > 0:
            self._analyze_soft_hard_gap()
            self._analyze_logit_sharpness()
            self._analyze_element_switching()
            self._analyze_pcfm_internals()
            self._analyze_anomaly_correlation()
        self._save_results()

    # ------------------------------------------------------------------
    # Existing analysis (convergence, histogram, per-sample)
    # ------------------------------------------------------------------

    def _analyze_charge_convergence(self):
        plot_charge_convergence(
            self.timesteps, self.hard_charges_all, self.output_dir,
            title="Predicted Clean Charge Convergence",
        )
        plot_charge_histogram(
            self.hard_charges_all[-1], self.output_dir,
            title="Final Charge Distribution",
        )
        if self.n_samples > 0:
            per_sample_hard = self._compute_per_sample_hard_charges()
            plot_charge_per_sample(
                self.timesteps, per_sample_hard, self.output_dir,
                title="Per-Sample Charge Trajectories",
            )

    def _compute_per_sample_hard_charges(self) -> np.ndarray:
        """Returns (n_steps, n_samples) hard charges from per-sample logits."""
        result = np.zeros((self.n_steps, self.n_samples))
        for si, logit_list in enumerate(self.sample_logits):
            for step_idx, logits in enumerate(logit_list):
                hard_emb = F.one_hot(
                    torch.argmax(logits, dim=-1), logits.shape[-1],
                ).float()
                result[step_idx, si] = self.charge_mod.per_atom_charge(hard_emb).sum().item()
        return result

    # ------------------------------------------------------------------
    # Soft-hard charge gap (H1)
    # ------------------------------------------------------------------

    def _analyze_soft_hard_gap(self):
        soft = np.zeros((self.n_samples, self.n_steps))
        hard = np.zeros((self.n_samples, self.n_steps))

        for si, logit_list in enumerate(self.sample_logits):
            for step_idx, logits in enumerate(logit_list):
                soft_emb = torch.softmax(logits / self.temperature, dim=-1)
                soft[si, step_idx] = self.charge_mod.per_atom_charge(soft_emb).sum().item()
                hard_emb = F.one_hot(
                    torch.argmax(logits, dim=-1), logits.shape[-1],
                ).float()
                hard[si, step_idx] = self.charge_mod.per_atom_charge(hard_emb).sum().item()

        self.soft_charges = soft
        self.hard_charges_ps = hard
        self._plot_charge_gap()

    # ------------------------------------------------------------------
    # Logit sharpness (H1 / H2)
    # ------------------------------------------------------------------

    def _analyze_logit_sharpness(self):
        ent_mean = np.zeros((self.n_samples, self.n_steps))
        ent_min = np.zeros((self.n_samples, self.n_steps))
        conf_mean = np.zeros((self.n_samples, self.n_steps))
        conf_min = np.zeros((self.n_samples, self.n_steps))

        for si, logit_list in enumerate(self.sample_logits):
            for step_idx, logits in enumerate(logit_list):
                p = torch.softmax(logits / self.temperature, dim=-1)
                entropy = -(p * torch.log(p + 1e-12)).sum(dim=-1)
                ent_mean[si, step_idx] = entropy.mean().item()
                ent_min[si, step_idx] = entropy.min().item()
                top2 = torch.topk(logits, 2, dim=-1).values
                gap = top2[:, 0] - top2[:, 1]
                conf_mean[si, step_idx] = gap.mean().item()
                conf_min[si, step_idx] = gap.min().item()

        self.entropy_mean = ent_mean
        self.entropy_min = ent_min
        self.confidence_mean = conf_mean
        self.confidence_min = conf_min
        self._plot_logit_sharpness()

    # ------------------------------------------------------------------
    # Element switching (H4)
    # ------------------------------------------------------------------

    def _analyze_element_switching(self):
        sf = np.zeros((self.n_samples, self.n_steps - 1))

        for si, logit_list in enumerate(self.sample_logits):
            prev_idx = torch.argmax(logit_list[0], dim=-1)
            for step_idx in range(1, len(logit_list)):
                cur_idx = torch.argmax(logit_list[step_idx], dim=-1)
                n_atoms = cur_idx.shape[0]
                switched = (cur_idx != prev_idx).sum().item()
                sf[si, step_idx - 1] = switched / n_atoms
                prev_idx = cur_idx

        self.switch_fractions = sf
        self._plot_element_switching()

    # ------------------------------------------------------------------
    # PCFM internals replay (H2)
    # ------------------------------------------------------------------

    def _analyze_pcfm_internals(self):
        scales = np.zeros((self.n_samples, self.n_steps))
        gnorms = np.zeros((self.n_samples, self.n_steps))
        cnorms = np.zeros((self.n_samples, self.n_steps))

        for si, logit_list in enumerate(self.sample_logits):
            for step_idx, logits in enumerate(logit_list):
                s, g, c = self._replay_pcfm_step(logits)
                scales[si, step_idx] = s
                gnorms[si, step_idx] = g
                cnorms[si, step_idx] = c

        self.pcfm_scales = scales
        self.grad_norms = gnorms
        self.correction_norms = cnorms
        self._plot_pcfm_internals()

    @torch.enable_grad()
    def _replay_pcfm_step(self, logits: torch.Tensor) -> tuple[float, float, float]:
        """Replay one decoupled PCFM step and return (|scale|, grad_norm_sq, correction_norm).

        Matches the actual pcfm_project implementation: Q from hard argmax,
        gradient from softmax at self.temperature.
        """
        # Hard charge (exact)
        hard_emb = F.one_hot(
            torch.argmax(logits, dim=-1), logits.shape[-1],
        ).float()
        batch_indices = torch.zeros(logits.shape[0], dtype=torch.long)
        Q_hard = self.charge_mod.batch_charge(hard_emb, batch_indices, 1)

        # Soft gradient at self.temperature
        theta = logits.detach().clone().requires_grad_(True)
        soft = torch.softmax(theta / self.temperature, dim=-1)
        Q_soft = self.charge_mod.batch_charge(soft, batch_indices, 1)
        Q_soft.sum().backward()
        grad = theta.grad

        grad_norm_sq = (grad * grad).sum().item()
        clamped = max(grad_norm_sq, 1e-12)
        scale = Q_hard.item() / clamped
        correction = abs(scale) * grad.detach().norm().item()

        return abs(scale), grad_norm_sq, correction

    # ------------------------------------------------------------------
    # Anomaly correlation (synthesis)
    # ------------------------------------------------------------------

    def _analyze_anomaly_correlation(self):
        if self.hard_charges_ps is None:
            return
        self._plot_anomaly_correlation()
        self._plot_worst_sample_diagnostics()

    def _plot_worst_sample_diagnostics(self):
        max_abs = np.abs(self.hard_charges_ps).max(axis=1)
        n_worst = min(5, self.n_samples)
        worst_indices = np.argsort(max_abs)[-n_worst:][::-1]

        for si in worst_indices:
            sample_data = {'hard_charge': self.hard_charges_ps[si]}
            if self.soft_charges is not None:
                sample_data['soft_charge'] = self.soft_charges[si]
            if self.pcfm_scales is not None:
                sample_data['pcfm_scale'] = self.pcfm_scales[si]
            if self.switch_fractions is not None:
                sample_data['switch_fraction'] = self.switch_fractions[si]
            if self.entropy_mean is not None:
                sample_data['entropy_mean'] = self.entropy_mean[si]
            self._plot_sample_diagnostic(sample_data, si)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_charge_gap(self):
        import matplotlib.pyplot as plt

        soft, hard = self.soft_charges, self.hard_charges_ps
        t = self.timesteps

        soft_mean, soft_std = soft.mean(axis=0), soft.std(axis=0)
        hard_mean, hard_std = hard.mean(axis=0), hard.std(axis=0)
        gap = np.abs(hard - soft)
        gap_mean, gap_std = gap.mean(axis=0), gap.std(axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax1.plot(t, hard_mean, color='#2563eb', label='Hard (argmax)')
        ax1.fill_between(t, hard_mean - hard_std, hard_mean + hard_std, color='#2563eb', alpha=0.2)
        ax1.plot(t, soft_mean, color='#dc2626', label='Soft (continuous)')
        ax1.fill_between(t, soft_mean - soft_std, soft_mean + soft_std, color='#dc2626', alpha=0.2)
        ax1.set_ylabel('Total Charge')
        ax1.set_title('Soft vs Hard Charge Gap')
        ax1.legend()

        ax2.plot(t, gap_mean, color='#7c3aed')
        ax2.fill_between(t, gap_mean - gap_std, gap_mean + gap_std, color='#7c3aed', alpha=0.2)
        ax2.set_xlabel('Timestep t')
        ax2.set_ylabel('|Q_hard - Q_soft|')
        ax2.set_title('Charge Gap')

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'charge_gap.pdf'))
        plt.close(fig)

    def _plot_logit_sharpness(self):
        import matplotlib.pyplot as plt

        t = self.timesteps
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        ent_m = self.entropy_mean.mean(axis=0)
        ent_m_std = self.entropy_mean.std(axis=0)
        ent_lo = self.entropy_min.mean(axis=0)
        ax1.plot(t, ent_m, color='#2563eb', label='Mean entropy (avg over samples)')
        ax1.fill_between(t, ent_m - ent_m_std, ent_m + ent_m_std, color='#2563eb', alpha=0.2)
        ax1.plot(t, ent_lo, color='#dc2626', linestyle='--', label='Min entropy (avg over samples)')
        ax1.set_ylabel('Softmax Entropy')
        ax1.set_title('Logit Sharpness')
        ax1.legend()

        conf_m = self.confidence_mean.mean(axis=0)
        conf_m_std = self.confidence_mean.std(axis=0)
        conf_lo = self.confidence_min.mean(axis=0)
        ax2.plot(t, conf_m, color='#2563eb', label='Mean confidence (avg over samples)')
        ax2.fill_between(t, conf_m - conf_m_std, conf_m + conf_m_std, color='#2563eb', alpha=0.2)
        ax2.plot(t, conf_lo, color='#dc2626', linestyle='--', label='Min confidence (avg over samples)')
        ax2.set_xlabel('Timestep t')
        ax2.set_ylabel('Top-1 minus Top-2 Logit')
        ax2.set_title('Logit Confidence')
        ax2.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'logit_sharpness.pdf'))
        plt.close(fig)

    def _plot_element_switching(self):
        import matplotlib.pyplot as plt

        t = self.timesteps[1:]
        mean = self.switch_fractions.mean(axis=0)
        std = self.switch_fractions.std(axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t, mean, color='#2563eb')
        ax.fill_between(t, mean - std, mean + std, color='#2563eb', alpha=0.2)
        ax.set_xlabel('Timestep t')
        ax.set_ylabel('Fraction of Atoms Switched')
        ax.set_title('Element Switching Rate')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'element_switching.pdf'))
        plt.close(fig)

    def _plot_pcfm_internals(self):
        import matplotlib.pyplot as plt

        t = self.timesteps
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

        for ax, data, ylabel, label in [
            (ax1, self.pcfm_scales, '|Q / |∇Q|²|', 'Scale factor'),
            (ax2, self.grad_norms, '|∇_θ Q|²', 'Gradient norm²'),
            (ax3, self.correction_norms, '‖correction‖', 'Correction norm'),
        ]:
            m = data.mean(axis=0)
            s = data.std(axis=0)
            ax.plot(t, m, color='#2563eb', label=label)
            ax.fill_between(t, m - s, m + s, color='#2563eb', alpha=0.2)
            worst_idx = int(np.argmax(data.max(axis=1)))
            ax.plot(t, data[worst_idx], color='#dc2626', alpha=0.5,
                    linewidth=0.8, label=f'Worst sample #{worst_idx}')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)

        ax3.set_xlabel('Timestep t')
        ax1.set_title('PCFM Projection Internals')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'pcfm_internals.pdf'))
        plt.close(fig)

    def _plot_anomaly_correlation(self, charge_threshold=5.0):
        import matplotlib.pyplot as plt

        spike_mag = np.abs(self.hard_charges_ps)
        mask = spike_mag > charge_threshold
        if not mask.any():
            return

        spike_vals = spike_mag[mask]

        panels: list[tuple[np.ndarray, str]] = []
        if self.pcfm_scales is not None:
            panels.append((self.pcfm_scales[mask], '|Scale factor|'))
        if self.grad_norms is not None:
            panels.append((self.grad_norms[mask], '|∇_θ Q|²'))
        if self.switch_fractions is not None:
            sf_padded = np.full_like(self.hard_charges_ps, np.nan)
            sf_padded[:, 1:] = self.switch_fractions
            panels.append((sf_padded[mask], 'Switch fraction'))
        if self.confidence_min is not None:
            panels.append((self.confidence_min[mask], 'Min confidence'))

        if not panels:
            return

        n_panels = len(panels)
        cols = min(n_panels, 2)
        rows = (n_panels + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

        for idx, (vals, ylabel) in enumerate(panels):
            ax = axes[idx // cols][idx % cols]
            valid = ~np.isnan(vals)
            ax.scatter(spike_vals[valid], vals[valid], alpha=0.4, s=10, color='#2563eb')
            ax.set_xlabel('|Q_hard| at spike')
            ax.set_ylabel(ylabel)

        for idx in range(n_panels, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle('Anomaly Correlation')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'anomaly_correlation.pdf'))
        plt.close(fig)

    def _plot_sample_diagnostic(self, sample_data: dict, sample_idx: int):
        import matplotlib.pyplot as plt

        t = self.timesteps
        fig, axes = plt.subplots(4, 1, figsize=(8, 14), sharex=True)

        axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        axes[0].plot(t, sample_data['hard_charge'], color='#2563eb', label='Hard')
        if 'soft_charge' in sample_data:
            axes[0].plot(t, sample_data['soft_charge'], color='#dc2626', label='Soft')
        axes[0].set_ylabel('Total Charge')
        axes[0].set_title(f'Sample #{sample_idx} Diagnostic')
        axes[0].legend(fontsize=8)

        if 'pcfm_scale' in sample_data:
            axes[1].plot(t, sample_data['pcfm_scale'], color='#7c3aed')
            axes[1].set_ylabel('|Scale factor|')
        else:
            axes[1].set_visible(False)

        if 'switch_fraction' in sample_data:
            sf = sample_data['switch_fraction']
            axes[2].plot(t[1:len(sf)+1], sf, color='#059669')
            axes[2].set_ylabel('Switch fraction')
        else:
            axes[2].set_visible(False)

        if 'entropy_mean' in sample_data:
            axes[3].plot(t, sample_data['entropy_mean'], color='#d97706')
            axes[3].set_ylabel('Mean entropy')
        else:
            axes[3].set_visible(False)

        axes[3].set_xlabel('Timestep t')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f'sample_{sample_idx:05d}_diagnostic.pdf'))
        plt.close(fig)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def _save_results(self):
        n_total = self.hard_charges_all.shape[1]
        final_all = self.hard_charges_all[-1]  # (n_total,)
        n_balanced = int((final_all == 0).sum())
        balanced_pct = n_balanced / max(n_total, 1) * 100

        metrics: dict = {
            "source_run_id": self.run.analyzer.source_run_id or self.run.id,
            "analysis_temperature": self.temperature,
            "n_total_samples": n_total,
            "n_individual_samples": self.n_samples,
            "n_balanced_samples": n_balanced,
            "balanced_pct": balanced_pct,
        }

        if self.hard_charges_ps is not None:
            final_hard = self.hard_charges_ps[:, -1]
            metrics["final_hard_charge_mean"] = float(final_hard.mean())
            metrics["final_hard_charge_std"] = float(final_hard.std())

        if self.soft_charges is not None and self.hard_charges_ps is not None:
            gap = np.abs(self.hard_charges_ps - self.soft_charges)
            metrics["charge_gap_mean"] = float(gap.mean())
            metrics["charge_gap_max"] = float(gap.max())

        if self.pcfm_scales is not None:
            metrics["pcfm_scale_max"] = float(self.pcfm_scales.max())
            metrics["pcfm_scale_mean"] = float(self.pcfm_scales.mean())

        if self.grad_norms is not None:
            metrics["grad_norm_sq_min"] = float(self.grad_norms.min())

        if self.switch_fractions is not None:
            metrics["switch_fraction_max"] = float(self.switch_fractions.max())

        self.run.results.append(ResultEntry(
            timestamp=datetime.now(timezone.utc),
            metrics=metrics,
            outputs={},
        ))
        save_run(self.run)


def analyze(run: Run):
    """Run charge analysis on saved trajectory data."""
    TrajectoryAnalyzer(run).main()
