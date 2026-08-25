from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class RadiusQuery:
    """Flattened exact-radius neighbours for a batch of query points."""

    indices: Tensor
    owners: Tensor
    nearest: Tensor
    counts: Tensor


class BackgroundSpatialIndex:
    """CPU cKDTree sidecar for a persistent static Gaussian background.

    DecoupleGS only queries local, finite-radius columns/probes.  Keeping the
    immutable means in a spatial sidecar avoids evaluating every background
    Gaussian for every vehicle while leaving the paper's weighting equations
    unchanged.  Candidate weights are still evaluated by torch on the source
    device; the tree is used only to reject points that provably have zero
    weight.
    """

    def __init__(self, means: Tensor) -> None:
        if means.ndim != 2 or means.shape[1] != 3 or means.shape[0] == 0:
            raise ValueError("background means must have shape [N, 3] and be non-empty")
        try:
            from scipy.spatial import cKDTree
        except ImportError as error:  # pragma: no cover - project environments include scipy.
            raise RuntimeError("indexed local queries require scipy") from error
        # cKDTree owns/normalizes this CPU snapshot. The simulation background
        # is persistent, so this transfer is a one-time scene preparation cost.
        self._means = np.asarray(
            means.detach().to(device="cpu", dtype=torch.float64).numpy(),
            dtype=np.float64,
            order="C",
        )
        self._tree_type = cKDTree
        self._trees: dict[tuple[int, ...], object] = {
            (0, 1, 2): cKDTree(self._means, compact_nodes=True, balanced_tree=True)
        }
        self._tree_lock = Lock()

    def __len__(self) -> int:
        return self._means.shape[0]

    def _tree(self, axes: tuple[int, ...]):
        if not axes or len(set(axes)) != len(axes) or any(axis not in (0, 1, 2) for axis in axes):
            raise ValueError("axes must be a non-empty unique subset of (0, 1, 2)")
        tree = self._trees.get(axes)
        if tree is not None:
            return tree
        # scipy queries are thread-safe; lazily constructing a new projected
        # tree is guarded because six camera workers may share this index.
        with self._tree_lock:
            tree = self._trees.get(axes)
            if tree is None:
                tree = self._tree_type(
                    self._means[:, axes], compact_nodes=True, balanced_tree=True
                )
                self._trees[axes] = tree
        return tree

    def query_radius(
        self,
        positions: Tensor,
        radius: float | Tensor | Sequence[float],
        *,
        axes: Sequence[int] = (0, 1, 2),
    ) -> RadiusQuery:
        axes_tuple = tuple(int(axis) for axis in axes)
        if positions.ndim == 1:
            positions = positions[None]
        if positions.ndim != 2 or positions.shape[1] != len(axes_tuple):
            raise ValueError("positions must have one column per indexed axis")
        query = np.asarray(
            positions.detach().to(device="cpu", dtype=torch.float64).numpy(),
            dtype=np.float64,
            order="C",
        )
        if isinstance(radius, Tensor):
            radius_np = np.asarray(radius.detach().to(device="cpu", dtype=torch.float64).numpy())
        else:
            radius_np = np.asarray(radius, dtype=np.float64)
        if radius_np.ndim > 1 or (
            radius_np.ndim == 1 and radius_np.shape != (positions.shape[0],)
        ):
            raise ValueError("radius must be scalar or contain one value per query")
        if bool((radius_np <= 0).any()):
            raise ValueError("radius must be positive")
        tree = self._tree(axes_tuple)
        neighbours = tree.query_ball_point(query, radius_np, return_sorted=True)
        # The nearest entry is the analytical zero-denominator fallback used by
        # sample_local_probe; compute it even for non-empty radius queries since
        # every candidate may have zero visibility.
        _, nearest = tree.query(query, k=1)
        counts_np = np.fromiter((len(row) for row in neighbours), dtype=np.int64)
        total = int(counts_np.sum())
        if total:
            indices_np = np.concatenate(
                [np.asarray(row, dtype=np.int64) for row in neighbours if row]
            )
            owners_np = np.repeat(np.arange(len(neighbours), dtype=np.int64), counts_np)
        else:
            indices_np = np.empty(0, dtype=np.int64)
            owners_np = np.empty(0, dtype=np.int64)
        device = positions.device
        return RadiusQuery(
            indices=torch.from_numpy(indices_np).to(device=device),
            owners=torch.from_numpy(owners_np).to(device=device),
            nearest=torch.as_tensor(nearest, dtype=torch.int64, device=device).reshape(-1),
            counts=torch.from_numpy(counts_np).to(device=device),
        )
