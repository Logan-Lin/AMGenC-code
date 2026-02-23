from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


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

    def __init__(self, element_embedding, config=None):
        self.element_embedding = element_embedding
        self.pos_path = MaterialFlowPath()
        self.el_path = FlowPath()

        # Optional categorical element distribution from config
        self.element_dist = None
        self.noise_sigma = 1.0
        if config is not None:
            if config.element_dist:
                probs = torch.tensor(config.element_dist, dtype=torch.float)
                self.element_dist = probs / probs.sum()
            if config.noise_sigma is not None:
                self.noise_sigma = config.noise_sigma

    def _sample_training_noise(self, clean_emb):
        """Sample training noise with OT coupling.

        If element_dist is set, centers noise at the true clean element embedding
        (OT coupling: reduces path crossings). Otherwise scaled Gaussian.
        """
        sigma = self.noise_sigma
        if self.element_dist is None:
            return sigma * torch.randn_like(clean_emb)
        return clean_emb + sigma * torch.randn_like(clean_emb)

    def _sample_inference_noise(self, like_tensor):
        """Sample inference noise from the marginal mixture.

        If element_dist is set, samples from a mixture of Gaussians centered at
        one-hot vectors with categorical mixing weights (matches training marginal).
        Otherwise scaled Gaussian.
        """
        sigma = self.noise_sigma
        if self.element_dist is None:
            return sigma * torch.randn_like(like_tensor)
        n_atoms = like_tensor.shape[0]
        n_elements = like_tensor.shape[1]
        dist = self.element_dist.to(like_tensor.device)
        indices = torch.multinomial(dist.expand(n_atoms, -1), 1).squeeze(1)
        onehot = F.one_hot(indices, n_elements).float()
        return onehot + sigma * torch.randn_like(like_tensor)

    def sample_t(self, batch) -> torch.FloatTensor:
        """Sample uniform t in [0, 1] with shape (batch_size,)."""
        return torch.rand(batch.get_batch_size(), device=batch.get_positions().device)

    def sample_source(self, sample):
        """Create source sample at t=0: uniform positions + Gaussian element embeddings.

        Args:
            sample: clean Sample/Batch to match structure (lattice, num_atoms, etc.)

        Returns:
            Source sample with randomized positions and Gaussian element embeddings.
        """
        source = sample.randomize_uniform()
        clean_emb = sample.get_element_emb()
        if clean_emb is None:
            clean_emb = self.element_embedding.embed(sample.get_elements())
        noise_emb = self._sample_inference_noise(clean_emb)
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
        clean_emb = clean_batch.get_element_emb()
        if clean_emb is None:
            clean_emb = self.element_embedding.embed(clean_batch.get_elements())
        noise_emb = self._sample_training_noise(clean_emb)
        v_el_target = self.el_path.velocity(noise_emb, clean_emb)
        flow_el = self.el_path.interpolate(noise_emb, clean_emb, t_atom)

        flow_batch = clean_batch.update_attrs(
            positions=flow_positions,
            element_emb=flow_el,
            elements=self.element_embedding.unembed(flow_el),
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

        cur_emb = sample.get_element_emb()
        new_el = cur_emb + dt * v_el
        clean_el = cur_emb + (1 - t_atom) * v_el

        next_sample = sample.update_attrs(
            positions=new_positions,
            element_emb=new_el,
            elements=self.element_embedding.unembed(new_el),
        )
        pred_clean = sample.update_attrs(
            positions=clean_positions,
            element_emb=clean_el,
            elements=self.element_embedding.unembed(clean_el),
        )

        return next_sample, pred_clean

    def generate(self, source_sample, n_steps, model, cond=None,
                 charge_module=None, pcfm_temperature=0.1):
        """Generate by integrating from t=0 to t=1.

        Args:
            source_sample: source Sample/Batch at t=0.
            n_steps: number of integration steps.
            model: velocity network.
            cond: optional condition tensor.
            charge_module: optional ChargeModule for PCFM projection.
            pcfm_temperature: softmax temperature for PCFM projection.

        Returns:
            (trajectory, pred_cleans): trajectory has length n_steps + 1,
            pred_cleans has length n_steps.
        """
        timesteps = np.linspace(0, 1, n_steps + 1)
        device = source_sample.get_positions().device
        batch_size = source_sample.get_batch_size()

        sample = source_sample
        trajectory = [sample]
        pred_cleans = []

        for i in range(n_steps):
            t_val = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]
            t_tensor = torch.full((batch_size,), t_val, dtype=torch.float, device=device)
            sample, pred_clean = self.integrate_step(sample, t_tensor, dt, model, cond=cond)

            # PCFM projection: correct element logits to push charge toward zero
            if charge_module is not None and t_val < 1.0:
                batch_indices = sample.get_batch_indices()
                clean_el = pred_clean.get_element_emb()
                corrected_el = charge_module.pcfm_project(
                    clean_el, batch_indices, batch_size, temperature=pcfm_temperature,
                )

                # Reverse update via OT displacement interpolant (PCFM Eq. 5):
                # ū_τ' = (1 - τ') u₀ + τ' u_proj
                # where u₀ is the source element embedding and u_proj is corrected_el.
                t_prime = timesteps[i + 1]
                source_el = source_sample.get_element_emb()
                new_el = (1 - t_prime) * source_el + t_prime * corrected_el

                sample = sample.update_attrs(
                    element_emb=new_el,
                    elements=self.element_embedding.unembed(new_el),
                )
                pred_clean = pred_clean.update_attrs(
                    element_emb=corrected_el,
                    elements=self.element_embedding.unembed(corrected_el),
                )

            trajectory.append(sample)
            pred_cleans.append(pred_clean)

        return trajectory, pred_cleans


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import MaterialDataset, Batch
    from db import Property
    from nn.layers import OneHotElementEmbedding

    elements = ["Si", "O", "Li", "Al", "Ba", "Be", "Ca", "K", "P", "Ti", "Zn"]
    props = [
        Property(name="E [GPa]", offset=75.6612144972, scale=16.3364790889),
        Property(name="G [GPa]", offset=30.9556123130, scale=6.4167374512),
    ]
    ds = MaterialDataset("data/bmp-sample", properties=props)

    el_emb = OneHotElementEmbedding(elements)
    fm = FlowMatcher(element_embedding=el_emb)

    batch = Batch([ds[0], ds[1]])
    batch = batch.update_attrs(element_emb=el_emb.embed(batch.get_elements()))

    # --- Shape smoke tests ---
    t = fm.sample_t(batch)
    flow_batch, (v_pos, v_el) = fm.compute_flow(batch, t)
    print(f"flow positions: {flow_batch.get_positions().shape}")
    print(f"v_pos target: {v_pos.shape}")
    print(f"v_el target: {v_el.shape}")

    source = fm.sample_source(batch)
    print(f"source positions: {source.get_positions().shape}")
    print(f"source element_emb: {source.get_element_emb().shape}")
    print(f"cond: {batch.cond}")

    atol = 1e-5

    # 1. FlowPath basic math
    fp = FlowPath()
    x0 = torch.randn(10, 3)
    x1 = torch.randn(10, 3)

    # Boundary: t=0 -> x0, t=1 -> x1
    assert torch.allclose(fp.interpolate(x0, x1, 0.0), x0, atol=atol), "FlowPath t=0 boundary"
    assert torch.allclose(fp.interpolate(x0, x1, 1.0), x1, atol=atol), "FlowPath t=1 boundary"

    # Velocity = x1 - x0
    assert torch.allclose(fp.velocity(x0, x1), x1 - x0, atol=atol), "FlowPath velocity"

    # Finite difference: (interp(t+eps) - interp(t)) / eps ≈ velocity
    eps = 0.1
    t_val = 0.3
    fd = (fp.interpolate(x0, x1, t_val + eps) - fp.interpolate(x0, x1, t_val)) / eps
    assert torch.allclose(fd, fp.velocity(x0, x1), atol=1e-3), "FlowPath finite diff"
    print("1. FlowPath basic math OK")

    # 2. MaterialFlowPath PBC-aware math
    mfp = MaterialFlowPath()
    s = ds[0]
    src = s.randomize_uniform()
    target_pos = s.get_positions()

    # Boundary t=0: interpolate at 0 = source positions
    interp0 = mfp.interpolate(src, target_pos, 0.0)
    assert torch.allclose(interp0, src.get_positions(), atol=atol), "MaterialFlowPath t=0"

    # Boundary t=1: source + 1*velocity should land on target (mod PBC)
    vel = mfp.velocity(src, target_pos)
    arrived = src.get_positions() + vel
    # Check via PBC: cal_velocity from arrived to target should be ~0
    arrived_sample = s.update_attrs(positions=arrived)
    residual = arrived_sample.cal_velocity(target_pos)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=atol), "MaterialFlowPath t=1 PBC"

    # Finite difference consistency
    t_val = 0.4
    eps_mfp = 0.1
    fd = (mfp.interpolate(src, target_pos, t_val + eps_mfp) - mfp.interpolate(src, target_pos, t_val)) / eps_mfp
    assert torch.allclose(fd, vel, atol=1e-3), "MaterialFlowPath finite diff"
    print("2. MaterialFlowPath PBC math OK")

    # 3. compute_flow boundary checks
    # At t=1: flow positions should recover clean positions (mod PBC)
    t_one = torch.ones(batch.get_batch_size())
    flow_b1, (vp1, ve1) = fm.compute_flow(batch, t_one)
    residual_pos = flow_b1.cal_velocity(batch.get_positions())
    assert torch.allclose(residual_pos, torch.zeros_like(residual_pos), atol=atol), "compute_flow t=1 positions"

    # At t=1: element flow should equal clean embeddings
    clean_emb = el_emb.embed(batch.get_elements())
    assert torch.allclose(flow_b1.get_element_emb(), clean_emb, atol=atol), "compute_flow t=1 elements"

    # At t=0: element flow should equal the noise (source)
    t_zero = torch.zeros(batch.get_batch_size())
    flow_b0, (vp0, ve0) = fm.compute_flow(batch, t_zero)
    # At t=0, flow_el = (1-0)*noise + 0*clean = noise, and v_el = clean - noise
    # So flow_el + v_el = clean
    assert torch.allclose(flow_b0.get_element_emb() + ve0, clean_emb, atol=atol), "compute_flow t=0 el consistency"
    print("3. compute_flow boundary checks OK")

    # 4. integrate_step Euler math
    # Mock model: constant velocity
    const_v_pos = torch.ones_like(batch.get_positions())
    const_v_el = torch.ones_like(el_emb.embed(batch.get_elements()))

    def mock_model(sample, t, cond=None):
        return const_v_pos, const_v_el

    source_sample = fm.sample_source(batch)
    t_cur = torch.full((batch.get_batch_size(),), 0.3)
    dt = 0.1
    next_s, pred_clean = fm.integrate_step(source_sample, t_cur, dt, mock_model)

    # Euler: new_pos = old_pos + dt * v
    expected_pos = source_sample.get_positions() + dt * const_v_pos
    assert torch.allclose(next_s.get_positions(), expected_pos, atol=atol), "Euler step positions"

    # Clean extrapolation: clean_pos = old_pos + (1 - t) * v
    batch_idx = source_sample.get_batch_indices()
    t_atom = t_cur[batch_idx].view(-1, 1)
    expected_clean_pos = source_sample.get_positions() + (1 - t_atom) * const_v_pos
    assert torch.allclose(pred_clean.get_positions(), expected_clean_pos, atol=atol), "Clean extrapolation positions"

    # Element Euler step
    cur_emb = source_sample.get_element_emb()
    expected_el = cur_emb + dt * const_v_el
    assert torch.allclose(next_s.get_element_emb(), expected_el, atol=atol), "Euler step elements"

    expected_clean_el = cur_emb + (1 - t_atom) * const_v_el
    assert torch.allclose(pred_clean.get_element_emb(), expected_clean_el, atol=atol), "Clean extrapolation elements"
    print("4. integrate_step Euler math OK")

    # 5. generate trajectory check
    n_steps = 5
    source_sample = fm.sample_source(batch)
    traj, pred_cleans = fm.generate(source_sample, n_steps, mock_model)

    assert len(traj) == n_steps + 1, f"Trajectory length: {len(traj)} != {n_steps + 1}"
    assert len(pred_cleans) == n_steps, f"Pred cleans length: {len(pred_cleans)} != {n_steps}"

    # With constant velocity v=1, integral from 0 to 1 gives displacement = 1*1 = 1
    # final_pos = source_pos + 1.0 * const_v_pos
    expected_final = source_sample.get_positions() + 1.0 * const_v_pos
    assert torch.allclose(traj[-1].get_positions(), expected_final, atol=1e-4), "Generate final positions"

    expected_final_el = source_sample.get_element_emb() + 1.0 * const_v_el
    assert torch.allclose(traj[-1].get_element_emb(), expected_final_el, atol=1e-4), "Generate final elements"
    print("5. generate trajectory OK")

    # 6. OT-coupled element noise sampling
    from db import FlowMatcherConfig

    # Heavy weight on O (index 1), light on others
    dist = [0.1, 0.6, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025, 0.0]
    cfg = FlowMatcherConfig(element_dist=dist)
    fm_cat = FlowMatcher(element_embedding=el_emb, config=cfg)

    assert fm_cat.element_dist is not None, "element_dist should be set"
    assert torch.allclose(fm_cat.element_dist.sum(), torch.tensor(1.0), atol=1e-5), "element_dist should be normalized"
    assert fm_cat.noise_sigma == 1.0, "default noise_sigma should be 1.0"

    # 6a. Training noise: OT-coupled (centered at clean embedding)
    clean_emb_test = el_emb.embed(batch.get_elements())
    training_noise = fm_cat._sample_training_noise(clean_emb_test)
    assert training_noise.shape == clean_emb_test.shape, "Training noise shape mismatch"

    # OT coupling: training noise = clean_emb + σ*ε, so the velocity target
    # v = clean_emb - noise = -σ*ε should have zero mean (no element switching)
    n_samples = 5000
    dummy_clean = el_emb.embed(batch.get_elements()[:1].expand(n_samples))  # repeat one atom
    training_samples = fm_cat._sample_training_noise(dummy_clean)
    v_target = dummy_clean - training_samples  # should be -σ*ε, mean ~0
    assert v_target.mean().abs() < 0.1, f"OT velocity target mean should be ~0, got {v_target.mean():.4f}"

    # 6b. Inference noise: marginal mixture (biased toward O)
    dummy = torch.randn(1000, el_emb.n_elements)
    inf_noise = fm_cat._sample_inference_noise(dummy)
    assert inf_noise.shape == dummy.shape, f"Shape mismatch: {inf_noise.shape} != {dummy.shape}"

    # Check that argmax distribution is biased toward O (index 1)
    decoded_indices = torch.argmax(inf_noise, dim=1)
    counts = torch.bincount(decoded_indices, minlength=el_emb.n_elements)
    o_fraction = counts[1].item() / 1000
    uniform_baseline = 1.0 / el_emb.n_elements
    assert o_fraction > uniform_baseline * 1.5, (
        f"Expected O fraction > {uniform_baseline * 1.5:.3f} (1.5x uniform), got {o_fraction}"
    )

    # 6c. noise_sigma scaling
    cfg_sigma = FlowMatcherConfig(element_dist=dist, noise_sigma=0.5)
    fm_sigma = FlowMatcher(element_embedding=el_emb, config=cfg_sigma)
    assert fm_sigma.noise_sigma == 0.5, "noise_sigma should be 0.5"

    # With smaller sigma, training noise should be closer to clean embedding
    train_noise_small = fm_sigma._sample_training_noise(dummy_clean)
    residual_small = (train_noise_small - dummy_clean).norm(dim=-1).mean()
    train_noise_large = fm_cat._sample_training_noise(dummy_clean)
    residual_large = (train_noise_large - dummy_clean).norm(dim=-1).mean()
    assert residual_small < residual_large, (
        f"Smaller sigma should produce tighter noise: σ=0.5 residual={residual_small:.3f} vs σ=1.0 residual={residual_large:.3f}"
    )

    # 6d. Without element_dist: both methods produce σ * randn
    fm_no_dist = FlowMatcher(element_embedding=el_emb)
    assert fm_no_dist.element_dist is None
    cfg_sigma_only = FlowMatcherConfig(noise_sigma=2.0)
    fm_sigma_only = FlowMatcher(element_embedding=el_emb, config=cfg_sigma_only)
    assert fm_sigma_only.element_dist is None
    assert fm_sigma_only.noise_sigma == 2.0

    # Verify sample_source and compute_flow work with categorical config
    source_cat = fm_cat.sample_source(batch)
    assert source_cat.get_element_emb().shape == el_emb.embed(batch.get_elements()).shape
    flow_cat, (vp_cat, ve_cat) = fm_cat.compute_flow(batch, t)
    assert flow_cat.get_element_emb().shape == el_emb.embed(batch.get_elements()).shape
    print("6. OT-coupled element noise OK")

    # 7. PCFM projection smoke test in generate()
    from nn.charge import create_charge_module

    charge_mod = create_charge_module("bmp", elements)
    n_steps_pcfm = 100
    tau_test = 0.2
    source_pcfm = fm.sample_source(batch)

    # Generate WITHOUT PCFM
    traj_no_pcfm, preds_no_pcfm = fm.generate(source_pcfm, n_steps_pcfm, mock_model)
    final_el_no = traj_no_pcfm[-1].get_element_emb()
    soft_no = torch.softmax(final_el_no / tau_test, dim=-1)
    Q_no = charge_mod.batch_charge(soft_no, traj_no_pcfm[-1].get_batch_indices(),
                                    traj_no_pcfm[-1].get_batch_size())

    # Generate WITH PCFM (same source for fair comparison)
    traj_pcfm, preds_pcfm = fm.generate(source_pcfm, n_steps_pcfm, mock_model,
                                         charge_module=charge_mod, pcfm_temperature=tau_test)
    final_el_yes = traj_pcfm[-1].get_element_emb()
    soft_yes = torch.softmax(final_el_yes / tau_test, dim=-1)
    Q_yes = charge_mod.batch_charge(soft_yes, traj_pcfm[-1].get_batch_indices(),
                                     traj_pcfm[-1].get_batch_size())

    # Hard charges: argmax → one-hot → charge
    hard_no = F.one_hot(final_el_no.argmax(dim=-1), final_el_no.shape[-1]).float()
    Q_no_hard = charge_mod.batch_charge(hard_no, traj_no_pcfm[-1].get_batch_indices(),
                                         traj_no_pcfm[-1].get_batch_size())
    hard_yes = F.one_hot(final_el_yes.argmax(dim=-1), final_el_yes.shape[-1]).float()
    Q_yes_hard = charge_mod.batch_charge(hard_yes, traj_pcfm[-1].get_batch_indices(),
                                          traj_pcfm[-1].get_batch_size())

    print(f"  Soft charges without PCFM: {Q_no.tolist()}")
    print(f"  Soft charges with PCFM:    {Q_yes.tolist()}")
    print(f"  Hard charges without PCFM: {Q_no_hard.tolist()}")
    print(f"  Hard charges with PCFM:    {Q_yes_hard.tolist()}")
    assert Q_yes.abs().mean() < Q_no.abs().mean(), \
        f"PCFM should reduce mean |charge|: without={Q_no.abs().mean():.4f}, with={Q_yes.abs().mean():.4f}"

    # Trajectory shape should be unchanged
    assert len(traj_pcfm) == n_steps_pcfm + 1
    assert len(preds_pcfm) == n_steps_pcfm

    # Positions should be identical (PCFM only affects elements)
    assert torch.allclose(traj_pcfm[-1].get_positions(), traj_no_pcfm[-1].get_positions(), atol=atol), \
        "PCFM should not affect positions"
    print("7. PCFM projection in generate() OK")

    print("\nAll flow matching math tests PASSED")
