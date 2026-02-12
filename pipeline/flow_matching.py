from __future__ import annotations

import numpy as np
import torch
from tqdm import trange


class FlowPath:
    """Standard linear interpolation flow path (optimal transport)."""

    def interpolate(self, x0, x1, t):
        """Interpolate between source x0 (noise) and target x1 (data) at time t."""
        return (1 - t) * x0 + t * x1

    def velocity(self, x0, x1):
        """Velocity field pointing from x0 to x1."""
        return x1 - x0


class MaterialFlowPath:
    """PBC-aware position flow path."""

    def velocity(self, source_batch, target_positions):
        """PBC-aware velocity from source positions to target positions.

        Args:
            source_batch: Sample/Batch with .positions and .cal_velocity().
            target_positions: (N, 3) target positions.

        Returns:
            (N, 3) velocity tensor.
        """
        return source_batch.cal_velocity(target_positions)

    def interpolate(self, source_batch, target_positions, t):
        """Interpolate: source_pos + t * velocity."""
        velocity = self.velocity(source_batch, target_positions)
        return source_batch.get_positions() + t * velocity


class FlowMatcher:
    """Flow matching for material generation.

    Convention: t=0 is noise, t=1 is data. Inference integrates from t=0 to t=1.
    """

    def __init__(self, element_embedding=None):
        self.element_embedding = element_embedding
        self.pos_path = MaterialFlowPath()
        self.el_path = FlowPath() if element_embedding is not None else None

    def sample_t(self, batch) -> torch.FloatTensor:
        """Sample uniform t in [0, 1] with shape (batch_size,)."""
        return torch.rand(batch.get_batch_size(), device=batch.get_positions().device)

    def sample_source(self, sample):
        """Create source sample at t=0: uniform positions + Gaussian element embeddings.

        Args:
            sample: clean Sample/Batch to match structure (lattice, num_atoms, etc.)

        Returns:
            Source sample with randomized positions and (optionally) Gaussian element embeddings.
        """
        source = sample.randomize_uniform()
        if self.element_embedding is not None:
            clean_emb = sample.get_element_emb()
            if clean_emb is None:
                clean_emb = self.element_embedding.embed(sample.get_elements())
            noise_emb = torch.randn_like(clean_emb)
            source = source.update_attrs(
                element_emb=noise_emb,
                elements=self.element_embedding.unembed(noise_emb),
            )
        return source

    def compute_flow(self, clean_batch, t):
        """Compute flow-matched interpolated sample and velocity targets.

        Args:
            clean_batch: clean data Sample/Batch (target at t=1).
            t: (B,) timestep tensor.

        Returns:
            (flow_batch, (v_pos_target, v_el_target))
        """
        batch_indices = clean_batch.get_batch_indices()
        t_shape = [-1] + [1] * (len(clean_batch.get_positions().shape) - 1)
        t_atom = t[batch_indices].view(t_shape)

        # Position flow
        source_batch = clean_batch.randomize_uniform()
        clean_positions = clean_batch.get_positions()
        v_pos_target = self.pos_path.velocity(source_batch, clean_positions)
        flow_positions = source_batch.get_positions() + t_atom * v_pos_target

        # Element flow
        flow_el = None
        v_el_target = None
        if self.el_path is not None:
            clean_emb = clean_batch.get_element_emb()
            if clean_emb is None:
                clean_emb = self.element_embedding.embed(clean_batch.get_elements())
            noise_emb = torch.randn_like(clean_emb)
            v_el_target = self.el_path.velocity(noise_emb, clean_emb)
            flow_el = self.el_path.interpolate(noise_emb, clean_emb, t_atom)

        flow_batch = clean_batch.update_attrs(
            positions=flow_positions,
            element_emb=flow_el,
            elements=self.element_embedding.unembed(flow_el) if flow_el is not None else None,
        )

        return flow_batch, (v_pos_target, v_el_target)

    def integrate_step(self, sample, t, dt, model, cond=None):
        """Single Euler integration step.

        Args:
            sample: current Sample/Batch at time t.
            t: (B,) current time.
            dt: scalar step size.
            model: velocity network.
            cond: optional condition tensor.

        Returns:
            (next_sample, pred_clean): stepped sample and clean extrapolation.
        """
        batch_indices = sample.get_batch_indices()
        t_shape = [-1] + [1] * (len(sample.get_positions().shape) - 1)
        t_atom = t[batch_indices].view(t_shape)

        v_pos, v_el = model(sample, t, cond=cond)

        # Euler: x_{t+dt} = x_t + dt * v_t
        new_positions = sample.get_positions() + dt * v_pos
        # Clean extrapolation: x_1 = x_t + (1 - t) * v_t
        clean_positions = sample.get_positions() + (1 - t_atom) * v_pos

        new_el, clean_el = None, None
        if v_el is not None and self.element_embedding is not None:
            cur_emb = sample.get_element_emb()
            new_el = cur_emb + dt * v_el
            clean_el = cur_emb + (1 - t_atom) * v_el

        next_sample = sample.update_attrs(
            positions=new_positions,
            element_emb=new_el,
            elements=self.element_embedding.unembed(new_el) if new_el is not None else None,
        )
        pred_clean = sample.update_attrs(
            positions=clean_positions,
            element_emb=clean_el,
            elements=self.element_embedding.unembed(clean_el) if clean_el is not None else None,
        )

        return next_sample, pred_clean

    def generate(self, source_sample, n_steps, model, cond=None):
        """Generate by integrating from t=0 to t=1.

        Args:
            source_sample: source Sample/Batch at t=0.
            n_steps: number of integration steps.
            model: velocity network.
            cond: optional condition tensor.

        Returns:
            List of samples along the trajectory (length n_steps + 1).
        """
        timesteps = np.linspace(0, 1, n_steps + 1)
        device = source_sample.get_positions().device
        batch_size = source_sample.get_batch_size()

        sample = source_sample
        trajectory = [sample]

        for i in trange(n_steps, desc="Generating", leave=False):
            t_val = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]
            t_tensor = torch.full((batch_size,), t_val, dtype=torch.float, device=device)
            sample, _ = self.integrate_step(sample, t_tensor, dt, model, cond=cond)
            trajectory.append(sample)

        return trajectory


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import Sample, Batch
    from nn.layers import OneHotElementEmbedding

    elements = ["Si", "O"]
    el_emb = OneHotElementEmbedding(elements)
    fm = FlowMatcher(element_embedding=el_emb)

    # Create a fake clean batch
    n_atoms = 10
    lattice = torch.eye(3) * 10.0
    samples = []
    for _ in range(2):
        s = Sample(
            elements=torch.randint(0, 2, (n_atoms,)) * 6 + 8,
            positions=torch.rand(n_atoms, 3) * 10.0,
            lattice=lattice.clone(),
        )
        s = s.update_attrs(element_emb=el_emb.embed(s.elements))
        samples.append(s)
    batch = Batch(samples)

    # Test compute_flow
    t = fm.sample_t(batch)
    flow_batch, (v_pos, v_el) = fm.compute_flow(batch, t)
    print(f"flow positions: {flow_batch.get_positions().shape}")
    print(f"v_pos target: {v_pos.shape}")
    print(f"v_el target: {v_el.shape}")

    # Test sample_source
    source = fm.sample_source(batch)
    print(f"source positions: {source.get_positions().shape}")
    print(f"source element_emb: {source.get_element_emb().shape}")

    print("Flow matching tests OK")
