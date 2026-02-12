import numpy as np
import torch
import ase
from ase.neighborlist import NeighborList as AseNeighborList

try:
    from torch_nl import compute_neighborlist
    _HAS_TORCH_NL = True
except ImportError:
    _HAS_TORCH_NL = False


def _positions_into_cell(pos: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Wrap positions into the unit cell."""
    invlat = torch.linalg.inv(cell)
    relpos = pos @ invlat
    relpos = relpos % 1.0
    return relpos @ cell


class Neighborlist:
    """Lazy neighbor list with optional padded build radius (init_r_cut).

    When init_r_cut is set, the list is built at init_r_cut but trimmed to
    r_cut on each query.  This avoids expensive rebuilds when atoms move
    small distances during diffusion steps.
    """

    def __init__(self, lattice, pbc, init_r_cut=None, use_torch_nl=True):
        self.lattice = lattice
        self.pbc = pbc
        self.init_r_cut = init_r_cut

        self.orig_pos = None   # positions from which current list was built
        self.r_cut = None      # r_cut with which it was built
        self.edges = None      # (rows, cols, offsets) tuple

        self.use_torch_nl = use_torch_nl and _HAS_TORCH_NL

    def set_init_r_cut(self, init_r_cut):
        self.init_r_cut = init_r_cut

    def to(self, device):
        nl = Neighborlist(self.lattice.to(device), self.pbc, self.init_r_cut, self.use_torch_nl)
        if self.orig_pos is not None:
            nl.orig_pos = self.orig_pos.to(device)
        nl.r_cut = self.r_cut
        if self.edges is not None:
            nl.edges = tuple(x.to(device) for x in self.edges)
        return nl

    @torch.no_grad()
    def update(self, positions, r_cut):
        """Rebuild the neighbor list from scratch at the given r_cut."""
        nat = positions.shape[0]
        if self.use_torch_nl:
            positions_c = _positions_into_cell(positions, self.lattice)
            neighbors, _batch_indices, offset_indices = compute_neighborlist(
                r_cut,
                positions_c,
                self.lattice,
                torch.tensor(self.pbc).to(positions.device),
                torch.zeros(nat, dtype=torch.long).to(positions.device),
                self_interaction=False,
            )
            rows = neighbors[0, :]
            cols = neighbors[1, :]
            d_cell = positions_c - positions
            offsets = offset_indices @ self.lattice - (d_cell[rows] - d_cell[cols])
        else:
            atoms = ase.Atoms(
                numbers=np.ones(nat, dtype=np.int32),
                positions=positions.detach().cpu().numpy(),
                cell=self.lattice.detach().cpu().numpy(),
                pbc=self.pbc,
            )
            nl = AseNeighborList(
                [r_cut / 2.0] * nat,
                self_interaction=False,
                bothways=False,
                skin=0.0,
            )
            nl.update(atoms)
            neighbors, offset_indices = nl.get_neighbors(slice(None))
            rows, cols, offsets = [], [], []
            lat = self.lattice.cpu()
            for i, (neis, offs) in enumerate(zip(neighbors, offset_indices)):
                rows.extend([i] * len(neis))
                cols.extend(neis)
                if self.lattice is None:
                    offsets.append(torch.zeros(len(neis), 2, dtype=positions.dtype))
                else:
                    offsets.append((torch.tensor(offs * 1.0).float() @ lat).to(positions.dtype))
            rows = torch.LongTensor(rows)
            cols = torch.LongTensor(cols)
            offsets = (
                torch.cat(offsets, dim=0) if len(offsets) > 0
                else torch.zeros(0, positions.shape[1])
            )
            rows = rows.to(positions.device)
            cols = cols.to(positions.device)
            offsets = offsets.to(positions.device)
            # Make edges bidirectional
            rows, cols, offsets = (
                torch.cat((rows, cols)),
                torch.cat((cols, rows)),
                torch.cat((offsets, -1 * offsets)),
            )

        self.r_cut = r_cut
        self.orig_pos = positions.clone()
        self.edges = (rows, cols, offsets)

    @torch.no_grad()
    def get_edges(self, positions, r_cut):
        """Return (src, dst, offsets) trimmed to r_cut, rebuilding if needed."""
        build_r_cut = self.init_r_cut if self.init_r_cut is not None else r_cut
        if self.edges is None:
            self.update(positions, build_r_cut)
        else:
            disp = torch.linalg.norm(positions - self.orig_pos, dim=1)
            d_max = torch.max(disp).detach().item()
            curr_r_cut = self.r_cut - 2 * d_max
            if curr_r_cut < r_cut:
                self.update(positions, build_r_cut)

        # Trim edges beyond r_cut using current positions
        row, col, offs = self.edges
        edge_len = torch.linalg.norm(positions[row, :] - positions[col, :] - offs, dim=1)
        keep_edge = edge_len <= r_cut
        return row[keep_edge], col[keep_edge], offs[keep_edge, :]
