from __future__ import annotations

import torch
from torch import nn

# BMP element map and charge tables (from BMP potential reference).
# Keys are 1-indexed BMP type IDs.

BMP_ELEMENT_MAP = {
     1: 'Si',
     2: 'O',
     3: 'Li',
     4: 'Na',
     5: 'K',
     6: 'Fe2+',
     7: 'Fe3+',
     8: 'Al',
     9: 'P',
    10: 'Ca',
    11: 'Be',
    12: 'Sr',
    13: 'Ba',
    14: 'Sc',
    15: 'Ti',
    16: 'Zr',
    17: 'Cr',
    18: 'Mn II',
    19: 'Mn III',
    20: 'Co',
    21: 'Ni',
    22: 'Cu I',
    23: 'Cu II',
    24: 'Ag',
    25: 'Zn',
    26: 'Ge',
    27: 'Sn',
    28: 'Nd',
    29: 'Gd',
    30: 'Er',
    31: 'Ga',
    32: 'Ce III',
    33: 'Ce IV',
    34: 'V IV',
    35: 'V V',
    36: 'Mg',
    37: 'Eu',
    38: 'B',
}

BMP_INV_ELEMENT_MAP = {v: k for k, v in BMP_ELEMENT_MAP.items()}

BMP_FORMAL_CHARGES = {
     1:  4,   2: -2,   3:  1,   4:  1,   5:  1,
     6:  2,   7:  3,   8:  3,   9:  5,  10:  2,
    11:  2,  12:  2,  13:  2,  14:  3,  15:  4,
    16:  4,  17:  3,  18:  2,  19:  3,  20:  2,
    21:  2,  22:  1,  23:  2,  24:  1,  25:  2,
    26:  4,  27:  4,  28:  3,  29:  3,  30:  3,
    31:  3,  32:  3,  33:  4,  34:  4,  35:  5,
    36:  2,  37:  3,  38:  3,
}


class ChargeModule(nn.Module):
    """Base class for computing per-atom and per-sample charges from element embeddings.

    Charge computation uses matrix multiplication: ``element_emb @ charge_vector``
    so that replacing hard one-hot vectors with ``softmax(x / t)`` makes the
    computation differentiable.

    Subclasses only need to implement ``_build_charge_vector``.
    """

    def __init__(self, elements: list[str]):
        super().__init__()
        self.n_elements = len(elements)
        charge_vector = self._build_charge_vector(elements)
        self.register_buffer('charge_vector', charge_vector)

    def _build_charge_vector(self, elements: list[str]) -> torch.Tensor:
        """Return a (n_elements,) tensor mapping one-hot index → charge value."""
        raise NotImplementedError

    def per_atom_charge(self, element_emb: torch.Tensor) -> torch.Tensor:
        """Compute per-atom charges.

        Args:
            element_emb: (N, n_elements) one-hot or soft element embeddings.

        Returns:
            (N,) per-atom charge values.
        """
        return element_emb @ self.charge_vector

    def sample_charge(self, element_emb: torch.Tensor) -> torch.Tensor:
        """Compute total charge for a single sample.

        Args:
            element_emb: (N, n_elements) element embeddings for one sample.

        Returns:
            Scalar tensor — sum of per-atom charges.
        """
        return self.per_atom_charge(element_emb).sum()

    def batch_charge(self, element_emb: torch.Tensor, batch_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Compute per-sample total charges for a batch (vectorized, no loops).

        Args:
            element_emb: (N_total, n_elements) element embeddings for all atoms.
            batch_indices: (N_total,) integer tensor assigning each atom to a sample.
            batch_size: Number of samples in the batch.

        Returns:
            (B,) tensor of per-sample total charges.
        """
        per_atom = self.per_atom_charge(element_emb)
        result = element_emb.new_zeros(batch_size)
        result.scatter_add_(0, batch_indices, per_atom)
        return result

    @torch.enable_grad()
    def pcfm_project(self, logits, batch_indices, batch_size, temperature=0.1):
        """PCFM Gauss-Newton projection to push total charge toward zero.

        Uses decoupled hard-Q / soft-gradient: Q is the exact discrete charge
        (from argmax one-hot), while ∇Q is computed via softmax at the given
        temperature for well-conditioned gradients.

        Applies: θ ← θ − (Q_hard / |∇_θ Q_soft|²) ∇_θ Q_soft

        Args:
            logits: (N_total, n_elements) element logits (pre-softmax).
            batch_indices: (N_total,) integer tensor assigning each atom to a sample.
            batch_size: Number of samples in the batch.
            temperature: Softmax temperature τ for gradient computation.

        Returns:
            (N_total, n_elements) corrected logits (detached).
        """
        # Exact hard charge (no gradient needed)
        hard_emb = torch.nn.functional.one_hot(
            torch.argmax(logits, dim=-1), logits.shape[-1],
        ).float()
        Q_hard = self.batch_charge(hard_emb, batch_indices, batch_size)

        # Soft gradient at the given temperature (well-conditioned)
        theta = logits.detach().clone().requires_grad_(True)
        soft = torch.softmax(theta / temperature, dim=-1)
        Q_soft = self.batch_charge(soft, batch_indices, batch_size)
        Q_soft.sum().backward()
        grad = theta.grad  # (N_total, n_elements)

        # Per-sample squared gradient norm
        grad_sq_per_atom = (grad * grad).sum(dim=-1)  # (N_total,)
        grad_norm_sq = logits.new_zeros(batch_size)
        grad_norm_sq.scatter_add_(0, batch_indices, grad_sq_per_atom)  # (batch_size,)

        # Clamp to avoid division by zero
        grad_norm_sq = grad_norm_sq.clamp(min=1e-12)

        # Scale uses hard Q (accurate) with soft gradient norm (well-conditioned)
        scale = Q_hard.detach() / grad_norm_sq  # (batch_size,)

        # Apply correction: theta - scale[batch_indices] * grad
        corrected = theta.detach() - scale[batch_indices].unsqueeze(-1) * grad.detach()
        return corrected

    def discrete_project(self, logits, batch_indices, batch_size):
        """Min-cost element re-assignment for exact charge balance via DP.

        Args:
            logits: (N_total, n_elements) element logits (from final x_1).
            batch_indices: (N_total,) sample assignment.
            batch_size: int.

        Returns:
            (N_total,) corrected element indices (long tensor).
        """
        raise NotImplementedError


class BMPChargeModule(ChargeModule):
    """Charge module using BMP formal charges."""

    def _build_charge_vector(self, elements: list[str]) -> torch.Tensor:
        v = torch.zeros(len(elements))
        for i, el in enumerate(elements):
            if el == "X":  # ghost atom, charge 0
                continue
            bmp_idx = BMP_INV_ELEMENT_MAP[el]
            v[i] = BMP_FORMAL_CHARGES[bmp_idx]
        return v

    def discrete_project(self, logits, batch_indices, batch_size):
        """Min-cost element re-assignment for exact charge balance via DP.

        Uses formal charges (integers) for clean DP states. For each sample,
        finds the minimum-cost set of element swaps to make formal charge == 0.
        Cost = logit gap sacrificed per swap (logits[i, current] - logits[i, new]).

        Args:
            logits: (N_total, n_elements) element logits (from final x_1).
            batch_indices: (N_total,) sample assignment.
            batch_size: int.

        Returns:
            (N_total,) corrected element indices (long tensor, same device as logits).
        """
        orig_device = logits.device
        # DP is sequential with small state tensors — run entirely on CPU
        logits = logits.cpu()
        batch_indices = batch_indices.cpu()
        fc = self.charge_vector.long().cpu()  # (m,) integer formal charges

        current = logits.argmax(dim=-1)  # (N_total,)
        result = current.clone()
        m = self.n_elements
        INF = float('inf')

        for b in range(batch_size):
            mask = batch_indices == b
            idx = mask.nonzero(as_tuple=True)[0]
            n = idx.shape[0]
            cur = current[idx]  # (n,) current element indices
            sample_logits = logits[idx]  # (n, m)

            # Compute current total formal charge
            Q = fc[cur].sum().item()
            if Q == 0:
                continue

            target_delta = -Q
            half_range = abs(target_delta) + 20
            # DP states: delta in [-half_range, half_range], offset by half_range
            n_states = 2 * half_range + 1
            states = torch.arange(n_states)  # (n_states,)

            cost = torch.full((n_states,), INF)
            cost[half_range] = 0.0  # delta=0 at start

            # Store choices for backtracking: all_choices[i, d] = element chosen
            all_choices = torch.empty(n, n_states, dtype=torch.long)

            for i in range(n):
                cur_el = cur[i].item()
                cur_fc = fc[cur_el].item()

                # Swap costs for all elements: (m,)
                swap_costs = sample_logits[i, cur_el] - sample_logits[i]
                swap_costs[cur_el] = 0.0

                # Charge deltas for each element: (m,)
                charge_deltas = fc - cur_fc  # (m,)

                # For each element j, compute source_indices[j, d] = d - charge_deltas[j]
                # This is the previous state that transitions to state d via element j
                source = states.unsqueeze(0) - charge_deltas.unsqueeze(1)  # (m, n_states)
                valid = (source >= 0) & (source < n_states)
                source_clamped = source.clamp(0, n_states - 1)

                # Look up previous costs and add swap costs
                shifted = cost[source_clamped]  # (m, n_states)
                shifted[~valid] = INF
                shifted = shifted + swap_costs.unsqueeze(1)  # (m, n_states)

                # Take minimum across elements
                cost, choice = shifted.min(dim=0)  # (n_states,) each
                all_choices[i] = choice

            # Find target delta index
            target_idx = half_range + target_delta
            if target_idx < 0 or target_idx >= n_states or cost[target_idx].item() == INF:
                # Fallback: find closest feasible delta
                feasible = cost < INF
                if feasible.any():
                    distances = (states[feasible] - half_range - target_delta).abs()
                    best_pos = distances.argmin()
                    target_idx = feasible.nonzero(as_tuple=True)[0][best_pos].item()
                else:
                    target_idx = half_range  # no change

            # Backtrack to recover element assignments
            d = target_idx
            for i in range(n - 1, -1, -1):
                chosen = all_choices[i, d].item()
                result[idx[i]] = chosen
                d = d - (fc[chosen].item() - fc[cur[i]].item())

        return result.to(orig_device)


CHARGE_MODULE_REGISTRY: dict[str, type[ChargeModule]] = {
    "bmp": BMPChargeModule,
}


def create_charge_module(name: str, elements: list[str]) -> ChargeModule:
    """Create a charge module by name."""
    if name not in CHARGE_MODULE_REGISTRY:
        raise ValueError(f"Unknown charge module: {name!r}. Available: {list(CHARGE_MODULE_REGISTRY.keys())}")
    return CHARGE_MODULE_REGISTRY[name](elements)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nn.layers import OneHotElementEmbedding
    from data import MaterialDataset, Batch

    elements = ["Si", "O", "Li", "Al", "Ba", "Be", "Ca", "K", "P", "Ti", "Zn"]
    emb = OneHotElementEmbedding(elements)
    charge_mod = create_charge_module("bmp", elements)

    print("charge_vector:", charge_mod.charge_vector)

    # Load sample data
    ds = MaterialDataset("data/bmp-sample")
    s0, s1 = ds[0], ds[1]

    # Single sample charge
    el_emb_0 = emb.embed(s0.elements)
    charge_0 = charge_mod.sample_charge(el_emb_0)
    print(f"\nSample 0: {s0.get_num_atoms()} atoms, charge = {charge_0.item():.4f}")

    # Batch charge (vectorized)
    batch = Batch([s0, s1])
    el_emb_batch = emb.embed(batch.elements)
    charges = charge_mod.batch_charge(el_emb_batch, batch.batch_indices, batch.get_batch_size())
    print(f"Batch charges: {charges}")

    # Verify batch[0] matches single sample
    assert torch.allclose(charges[0], charge_0, atol=0.01), \
        f"Batch charge[0] ({charges[0].item()}) should be close to single sample charge ({charge_0.item()})"
    print("Batch vs single sample: OK")

    # Test gradient flow with soft element embeddings
    soft_emb = torch.randn(s0.get_num_atoms(), len(elements), requires_grad=True)
    soft_emb_sm = torch.softmax(soft_emb / 0.1, dim=-1)
    soft_charge = charge_mod.sample_charge(soft_emb_sm)
    soft_charge.backward()
    assert soft_emb.grad is not None, "Gradient should flow through soft embeddings"
    print(f"Soft charge = {soft_charge.item():.4f}, grad norm = {soft_emb.grad.norm().item():.4f}")
    print("Gradient flow: OK")

    # Test PCFM projection
    print("\n--- PCFM Projection Test ---")
    n_atoms_pcfm = 500
    tau_test = 0.15
    n_pcfm_iter = 100
    logits_pcfm = torch.randn(n_atoms_pcfm, len(elements))
    batch_indices_pcfm = torch.zeros(n_atoms_pcfm, dtype=torch.long)  # single sample

    # Charge before projection
    soft_before = torch.softmax(logits_pcfm / tau_test, dim=-1)
    Q_before = charge_mod.sample_charge(soft_before).item()
    print(f"Charge before PCFM: {Q_before:.4f}")

    # Apply multiple PCFM iterations to drive charge close to zero
    corrected = logits_pcfm
    for iteration in range(n_pcfm_iter):
        corrected = charge_mod.pcfm_project(corrected, batch_indices_pcfm, 1, temperature=tau_test)

    soft_after = torch.softmax(corrected / tau_test, dim=-1)
    Q_after = charge_mod.sample_charge(soft_after).item()
    print(f"Charge after {n_pcfm_iter} PCFM iterations: {Q_after:.4f}")
    assert abs(Q_after) < abs(Q_before), \
        f"PCFM should reduce |charge|: before={abs(Q_before):.4f}, after={abs(Q_after):.4f}"

    # Test with batched input (2 samples)
    logits_b = torch.randn(80, len(elements))
    bi_b = torch.cat([torch.zeros(40, dtype=torch.long), torch.ones(40, dtype=torch.long)])
    soft_b_before = torch.softmax(logits_b / tau_test, dim=-1)
    Q_b_before = charge_mod.batch_charge(soft_b_before, bi_b, 2)
    corrected_b = logits_b
    for _ in range(n_pcfm_iter):
        corrected_b = charge_mod.pcfm_project(corrected_b, bi_b, 2, temperature=tau_test)
    soft_b_after = torch.softmax(corrected_b / tau_test, dim=-1)
    Q_b_after = charge_mod.batch_charge(soft_b_after, bi_b, 2)
    print(f"Batch charges before: {Q_b_before.tolist()}")
    print(f"Batch charges after:  {Q_b_after.tolist()}")
    assert (Q_b_after.abs() < Q_b_before.abs()).all(), "PCFM should reduce |charge| for all samples in batch"
    print("PCFM projection: OK")

    # Test discrete_project
    print("\n--- Discrete Projection Tests ---")
    fc = charge_mod.charge_vector

    # Test 1: Q != 0 → corrected to 0
    n_dp = 20
    logits_dp = torch.zeros(n_dp, len(elements))
    logits_dp[:, 0] = 10.0  # all Si (fc=4), Q = 20*4 = 80
    bi_dp = torch.zeros(n_dp, dtype=torch.long)
    corrected_idx = charge_mod.discrete_project(logits_dp, bi_dp, 1)
    corrected_fc = fc[corrected_idx]
    Q_corrected = corrected_fc.sum().item()
    assert Q_corrected == 0, f"DP should correct Q to 0, got {Q_corrected}"
    print(f"  Test 1 (Q!=0 → 0): PASS (Q: 80 → {Q_corrected})")

    # Test 2: Q == 0 → no swaps
    # Build a balanced composition: 2 Si (fc=4 each) + 4 O (fc=-2 each) = 8 - 8 = 0
    n_bal = 6
    logits_bal = torch.zeros(n_bal, len(elements))
    logits_bal[:2, 0] = 10.0  # Si
    logits_bal[2:, 1] = 10.0  # O
    bi_bal = torch.zeros(n_bal, dtype=torch.long)
    corrected_bal = charge_mod.discrete_project(logits_bal, bi_bal, 1)
    expected_bal = logits_bal.argmax(dim=-1)
    assert torch.equal(corrected_bal, expected_bal), \
        f"Balanced input should not be modified: {corrected_bal.tolist()} != {expected_bal.tolist()}"
    print("  Test 2 (Q==0 no swap): PASS")

    # Test 3: Batched input — two samples with different nonzero Q
    logits_batch_dp = torch.zeros(10, len(elements))
    logits_batch_dp[:5, 0] = 10.0  # sample 0: 5 Si, Q=20
    logits_batch_dp[5:, 0] = 10.0  # sample 1: 5 Si, Q=20
    bi_batch_dp = torch.cat([torch.zeros(5, dtype=torch.long), torch.ones(5, dtype=torch.long)])
    corrected_batch_dp = charge_mod.discrete_project(logits_batch_dp, bi_batch_dp, 2)
    for s in range(2):
        s_mask = bi_batch_dp == s
        Q_s = fc[corrected_batch_dp[s_mask]].sum().item()
        assert Q_s == 0, f"Sample {s} should have Q=0, got {Q_s}"
    print("  Test 3 (batched): PASS")

    # Test 4: Confidence-aware swaps — low-confidence atom swapped first
    n_conf = 5
    logits_conf = torch.zeros(n_conf, len(elements))
    # All atoms "confident" Si (fc=4) with large margin
    logits_conf[:, 0] = 10.0   # Si
    logits_conf[:, 1] = 0.0    # O (fc=-2)
    # Make atom 2 low-confidence (tiny margin)
    logits_conf[2, 0] = 1.0
    logits_conf[2, 1] = 0.9   # gap = 0.1
    # Q = 5*4 = 20, need to swap some atoms to O to balance
    bi_conf = torch.zeros(n_conf, dtype=torch.long)
    corrected_conf = charge_mod.discrete_project(logits_conf, bi_conf, 1)
    # Atom 2 should be swapped (lowest cost)
    original_conf = logits_conf.argmax(dim=-1)
    swapped_mask = corrected_conf != original_conf
    if swapped_mask.any():
        assert swapped_mask[2].item(), \
            f"Low-confidence atom 2 should be swapped, but swapped atoms: {swapped_mask.nonzero().flatten().tolist()}"
    print("  Test 4 (confidence-aware): PASS")

    # Test 5: Optimality check — small system with known optimal solution
    # 3 atoms: 2 Si (fc=4) + 1 O (fc=-2) = Q=6
    # Need delta=-6. Options: swap Si→O gives delta=-6 (one swap suffices)
    # Atom 0: Si with logit 10 (high confidence) — cost to swap = 10
    # Atom 1: Si with logit 3 (low confidence, O logit=2) — cost to swap = 1
    # Atom 2: O with logit 10 — no useful swap
    # Optimal: swap atom 1 to O, cost = 1
    logits_opt = torch.zeros(3, len(elements))
    logits_opt[0, 0] = 10.0   # Si (confident)
    logits_opt[1, 0] = 3.0    # Si (less confident)
    logits_opt[1, 1] = 2.0    # O logit = 2
    logits_opt[2, 1] = 10.0   # O (confident)
    bi_opt = torch.zeros(3, dtype=torch.long)
    corrected_opt = charge_mod.discrete_project(logits_opt, bi_opt, 1)
    Q_opt = fc[corrected_opt].sum().item()
    assert Q_opt == 0, f"Optimal test should have Q=0, got {Q_opt}"
    # Verify atom 1 was swapped (lowest cost swap)
    assert corrected_opt[1].item() != 0, "Atom 1 should be swapped from Si"
    assert corrected_opt[0].item() == 0, "Atom 0 (confident Si) should stay"
    assert corrected_opt[2].item() == 1, "Atom 2 (confident O) should stay"
    # Verify total cost is optimal: atom 1 Si→O costs 3-2 = 1
    original_opt = logits_opt.argmax(dim=-1)
    total_cost = 0.0
    for i in range(3):
        if corrected_opt[i] != original_opt[i]:
            total_cost += logits_opt[i, original_opt[i]].item() - logits_opt[i, corrected_opt[i]].item()
    assert total_cost <= 1.0 + 1e-6, f"Expected optimal cost <= 1.0, got {total_cost}"
    print(f"  Test 5 (optimality): PASS (cost={total_cost:.4f})")

    print("Discrete projection: OK")
