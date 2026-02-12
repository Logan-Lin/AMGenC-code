from __future__ import annotations

import torch
from torch import nn

from .layers import EGNN, OneHotElementEmbedding


class EgnnVelocityNet(nn.Module):
    """EGNN-based model predicting velocity fields for positions and elements."""

    def __init__(
        self,
        r_cut: float,
        elements: list[str],
        d_cond: int = 0,
        d_cond_embed: int = 0,
        hidden_nf: int = 128,
        n_layers: int = 4,
        n_coords: int = 1,
        coords_range: float = 15.0,
        normalization_factor: int = 100,
        tanh: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.r_cut = r_cut
        self.element_embedding = OneHotElementEmbedding(elements)
        n_elements = self.element_embedding.n_elements

        # Condition projection
        if d_cond > 0 and d_cond_embed > 0:
            self.cond_proj = nn.Linear(d_cond, d_cond_embed)
            cond_dim = d_cond_embed
        else:
            self.cond_proj = None
            cond_dim = d_cond

        # Input node features: time(1) + element_emb(n_elements) + cond(cond_dim)
        in_node_nf = 1 + n_elements + cond_dim

        self.egnn = EGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            r_cut=r_cut,
            out_node_nf=n_elements,
            n_layers=n_layers,
            n_coords=n_coords,
            coords_range=coords_range,
            normalization_factor=normalization_factor,
            tanh=tanh,
            residual=residual,
        )

    def forward(self, batch, t, cond=None):
        """Forward pass predicting velocity fields.

        Args:
            batch: Sample or Batch with positions and element_emb.
            t: (B,) timestep tensor.
            cond: (B, d_cond) optional condition tensor.

        Returns:
            (v_pos, v_el): position velocity (N, 3) and element velocity (N, n_elements).
        """
        positions = batch.get_positions()
        element_emb = batch.get_element_emb()
        batch_indices = batch.get_batch_indices()
        edges = batch.get_edges(self.r_cut)

        # Build per-atom time feature
        t_atom = t[batch_indices].unsqueeze(-1)  # (N, 1)

        # Build node features
        h_parts = [t_atom, element_emb]

        if cond is not None:
            if self.cond_proj is not None:
                cond = self.cond_proj(cond)
            cond_atom = cond[batch_indices]  # (N, cond_dim)
            h_parts.append(cond_atom)

        h = torch.cat(h_parts, dim=-1)  # (N, in_node_nf)

        h_out, x_out = self.egnn(h, positions, edges)

        # Position velocity: residual between input and output coordinates, mean-removed
        v_pos = positions - x_out[:, 0, :]
        v_pos = batch.remove_mean(v_pos)

        # Element velocity: node output
        v_el = h_out

        return v_pos, v_el


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import Sample, Batch

    elements = ["Si", "O"]
    model = EgnnVelocityNet(
        r_cut=5.0, elements=elements, hidden_nf=32, n_layers=2,
    )

    # Create fake samples
    n_atoms = 10
    lattice = torch.eye(3) * 10.0
    samples = []
    for _ in range(2):
        s = Sample(
            elements=torch.randint(0, 2, (n_atoms,)) * 6 + 8,  # O=8 or Si=14
            positions=torch.rand(n_atoms, 3) * 10.0,
            lattice=lattice.clone(),
        )
        s = s.update_attrs(element_emb=model.element_embedding.embed(s.elements))
        samples.append(s)

    batch = Batch(samples)
    t = torch.rand(2)
    v_pos, v_el = model(batch, t)
    print(f"v_pos shape: {v_pos.shape}")  # (20, 3)
    print(f"v_el shape: {v_el.shape}")    # (20, 2)
    print("Forward pass OK")
