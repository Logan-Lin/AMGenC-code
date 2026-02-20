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

BMP_CHARGES = {
     1:  2.4,   2: -1.2,   3:  0.6,   4:  0.6,   5:  0.6,
     6:  1.2,   7:  1.8,   8:  1.8,   9:  3.0,  10:  1.2,
    11:  1.2,  12:  1.2,  13:  1.2,  14:  1.8,  15:  2.4,
    16:  2.4,  17:  1.8,  18:  1.2,  19:  1.8,  20:  1.2,
    21:  1.2,  22:  0.6,  23:  1.2,  24:  0.6,  25:  1.2,
    26:  2.4,  27:  2.4,  28:  1.8,  29:  1.8,  30:  1.8,
    31:  1.8,  32:  1.8,  33:  2.4,  34:  2.4,  35:  3.0,
    36:  1.2,  37:  1.8,  38:  1.8,
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

        Applies one step of: θ ← θ − (Q / |∇_θ Q|²) ∇_θ Q
        where Q = 1ᵀ softmax(θ/τ) C, independently per sample.

        Args:
            logits: (N_total, n_elements) element logits (pre-softmax).
            batch_indices: (N_total,) integer tensor assigning each atom to a sample.
            batch_size: Number of samples in the batch.
            temperature: Softmax temperature τ.

        Returns:
            (N_total, n_elements) corrected logits (detached).
        """
        theta = logits.detach().clone().requires_grad_(True)
        soft = torch.softmax(theta / temperature, dim=-1)
        Q = self.batch_charge(soft, batch_indices, batch_size)
        Q.sum().backward()
        grad = theta.grad  # (N_total, n_elements)

        # Per-sample squared gradient norm: sum of grad^2 over all atoms and elements per sample
        grad_sq_per_atom = (grad * grad).sum(dim=-1)  # (N_total,)
        grad_norm_sq = logits.new_zeros(batch_size)
        grad_norm_sq.scatter_add_(0, batch_indices, grad_sq_per_atom)  # (batch_size,)

        # Clamp to avoid division by zero
        grad_norm_sq = grad_norm_sq.clamp(min=1e-12)

        # Per-sample scale: Q_i / |grad_i|^2
        scale = Q.detach() / grad_norm_sq  # (batch_size,)

        # Apply correction: theta - scale[batch_indices] * grad
        corrected = theta.detach() - scale[batch_indices].unsqueeze(-1) * grad.detach()
        return corrected


class BMPChargeModule(ChargeModule):
    """Charge module using BMP potential partial charges.

    Also provides formal charge computation via a separate buffer.
    """

    def __init__(self, elements: list[str]):
        super().__init__(elements)
        formal_charge_vector = self._build_formal_charge_vector(elements)
        self.register_buffer('formal_charge_vector', formal_charge_vector)

    def _build_charge_vector(self, elements: list[str]) -> torch.Tensor:
        v = torch.zeros(len(elements))
        for i, el in enumerate(elements):
            bmp_idx = BMP_INV_ELEMENT_MAP[el]
            v[i] = BMP_CHARGES[bmp_idx]
        return v

    def _build_formal_charge_vector(self, elements: list[str]) -> torch.Tensor:
        v = torch.zeros(len(elements))
        for i, el in enumerate(elements):
            bmp_idx = BMP_INV_ELEMENT_MAP[el]
            v[i] = BMP_FORMAL_CHARGES[bmp_idx]
        return v

    def per_atom_formal_charge(self, element_emb: torch.Tensor) -> torch.Tensor:
        """(N, n_elements) → (N,) per-atom formal charges."""
        return element_emb @ self.formal_charge_vector

    def batch_formal_charge(self, element_emb: torch.Tensor, batch_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """(N_total, n_elements), (N_total,), int → (B,) per-sample formal charges."""
        per_atom = self.per_atom_formal_charge(element_emb)
        result = element_emb.new_zeros(batch_size)
        result.scatter_add_(0, batch_indices, per_atom)
        return result


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
    print("formal_charge_vector:", charge_mod.formal_charge_vector)

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
    n_atoms_pcfm = 50
    tau_test = 0.5
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
    assert abs(Q_after) < 1.0, f"After {n_pcfm_iter} iterations, charge should be near zero, got {Q_after:.4f}"

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
