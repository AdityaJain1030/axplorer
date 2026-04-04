"""
K4-free graph environment for Axplorer.

Goal: find K4-free graphs on N vertices that minimise the independence number
alpha(G) relative to the bound f(d) = n * log(d) / d  (c = 1).

Score = f(d)/alpha, so lower is better
score = -1 is reserved for INVALID (K4-containing) graphs, matching cycle.py.

alpha is computed two ways controlled by --alpha_mode:
  exact  : branch-and-bound MIS (correct; feasible for N <= ~20)
  approx : greedy + random restarts (fast lower bound on alpha; optimistic —
           confirm any apparent counterexample with exact mode)
"""

import math
import random

import numpy as np

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import DenseTokenizer, SparseTokenizerSequenceKTokens, SparseTokenizerSingleInteger
from src.envs.utils import random_symmetry_adj_matrix, sort_graph_based_on_degree
from src.utils import bool_flag


# ─────────────────────────────────────────────────────────────────────────────
# Independence number solvers
# ─────────────────────────────────────────────────────────────────────────────

def _alpha_exact(adj: np.ndarray) -> int:
    """Branch-and-bound exact independence number. Feasible for N <= ~20."""
    N = adj.shape[0]
    nbr = [0] * N
    for i in range(N):
        for j in range(N):
            if adj[i, j]:
                nbr[i] |= 1 << j

    best = [0]

    def bb(candidates: int, size: int):
        if candidates == 0:
            if size > best[0]:
                best[0] = size
            return
        if size + bin(candidates).count('1') <= best[0]:
            return
        v = (candidates & -candidates).bit_length() - 1
        bb(candidates & ~nbr[v] & ~(1 << v), size + 1)
        bb(candidates & ~(1 << v), size)

    bb((1 << N) - 1, 0)
    return best[0]


def _alpha_approx(adj: np.ndarray, n_restarts: int = 30) -> int:
    """
    Greedy independent set with random restarts.
    Returns a lower bound on alpha (the largest IS found across restarts).
    Scores using this are optimistic: approx_alpha <= true_alpha, so
    f(d) - approx_alpha >= f(d) - true_alpha.
    Confirm any apparent counterexample with --alpha_mode exact.
    """
    N = adj.shape[0]
    nbr = [0] * N
    for i in range(N):
        for j in range(N):
            if adj[i, j]:
                nbr[i] |= 1 << j

    best = 0
    vertices = list(range(N))
    for _ in range(n_restarts):
        random.shuffle(vertices)
        available = (1 << N) - 1
        size = 0
        for v in vertices:
            if available & (1 << v):
                size += 1
                available &= ~nbr[v] & ~(1 << v)
        if size > best:
            best = size
    return best


# ─────────────────────────────────────────────────────────────────────────────
# K4 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_nbr_masks(adj: np.ndarray):
    N = adj.shape[0]
    nbr = [0] * N
    for i in range(N):
        for j in range(N):
            if adj[i, j]:
                nbr[i] |= 1 << j
    return nbr


def _has_k4(adj: np.ndarray) -> bool:
    N = adj.shape[0]
    nbr = _build_nbr_masks(adj)
    for a in range(N):
        for b in range(a + 1, N):
            if not (nbr[a] >> b & 1):
                continue
            c_bits = nbr[a] & nbr[b] & ~((1 << (b + 1)) - 1)
            while c_bits:
                lsb = c_bits & -c_bits
                c = lsb.bit_length() - 1
                c_bits ^= lsb
                if nbr[a] & nbr[b] & nbr[c] & ~((1 << (c + 1)) - 1):
                    return True
    return False


def _find_k4s(adj: np.ndarray):
    """Return list of all K4s as sorted 4-tuples of vertex indices."""
    N = adj.shape[0]
    nbr = _build_nbr_masks(adj)
    k4s = []
    for a in range(N):
        for b in range(a + 1, N):
            if not (nbr[a] >> b & 1):
                continue
            c_bits = nbr[a] & nbr[b] & ~((1 << (b + 1)) - 1)
            while c_bits:
                lsb_c = c_bits & -c_bits
                c = lsb_c.bit_length() - 1
                c_bits ^= lsb_c
                d_bits = nbr[a] & nbr[b] & nbr[c] & ~((1 << (c + 1)) - 1)
                while d_bits:
                    lsb_d = d_bits & -d_bits
                    d = lsb_d.bit_length() - 1
                    d_bits ^= lsb_d
                    k4s.append((a, b, c, d))
    return k4s


def _k4_edge_set(clique):
    a, b, c, d = clique
    return {(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)}


# ─────────────────────────────────────────────────────────────────────────────
# DataPoint
# ─────────────────────────────────────────────────────────────────────────────

class K4FreeDataPoint(DataPoint):
    MAKE_OBJECT_CANONICAL = False
    ALPHA_MODE = "exact"
    APPROX_RESTARTS = 30

    def __init__(self, N, init=False):
        super().__init__()
        self.N = N
        self.data = np.zeros((N, N), dtype=np.uint8)
        self.k4s = []

        if init:
            self._add_edges_greedily()
            if self.MAKE_OBJECT_CANONICAL:
                self.data = sort_graph_based_on_degree(self.data)
            self.calc_features()
            self.calc_score()

    # ── scoring ───────────────────────────────────────────────────────────────

    def _compute_alpha(self) -> int:
        if self.ALPHA_MODE == "exact":
            return _alpha_exact(self.data)
        return _alpha_approx(self.data, n_restarts=self.APPROX_RESTARTS)

    def _compute_f(self) -> float:
        """f(d) = n * log(d) / d."""
        d = int(self.data.sum(axis=1).max())
        if d <= 1:
            return 0.0
        return self.N * math.log(d) / d

    def calc_score(self):
        if self.k4s:
            self.score = -1
            return

        alpha = self._compute_alpha()
        f_val = self._compute_f()

        if alpha == 0:
            self.score = 0
            return
        
        # def truncate(number, decimals=0):
        #     factor = 10 ** decimals
        #     return math.trunc(number * factor) / factor
        # degrees = self.data.sum(axis=1)
        # d_mean = degrees.mean()
        # regularity = 1.0 / (1.0 + degrees.var())
        # self.score = (f_val / alpha) * regularity
        self.score = f_val / alpha

        # if f_val > alpha:
            # from logging import getLogger
            # getLogger().warning(
                # f"*** POTENTIAL COUNTEREXAMPLE: N={self.N}, alpha={alpha}, "
                # f"f(d)={f_val:.4f}, f(d)/alpha={f_val/alpha:.4f} ***"
            # )
    
    def calc_features(self):
        w = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                w.append(self.data[i, j])
        self.features = ",".join(map(str, w))

    # ── greedy construction ───────────────────────────────────────────────────

    def _add_edges_greedily(self):
        """Add edges in random order, skipping any that would create a K4."""
        np.random.seed(None)
        candidates = [
            (i, j) for i in range(self.N) for j in range(i + 1, self.N)
            if self.data[i, j] == 0
        ]
        np.random.shuffle(candidates)
        for i, j in candidates:
            if self.data[i, j] == 0:
                self.data[i, j] = 1
                self.data[j, i] = 1
                if _has_k4(self.data):
                    self.data[i, j] = 0
                    self.data[j, i] = 0

    # ── K4 removal ────────────────────────────────────────────────────────────

    def _remove_edges_greedily(self):
        """Remove the edge in the most K4s, repeat until K4-free."""
        while self.k4s:
            edge_count = {}
            for clique in self.k4s:
                for e in _k4_edge_set(clique):
                    edge_count[e] = edge_count.get(e, 0) + 1
            i, j = max(edge_count, key=edge_count.get)
            self.data[i, j] = 0
            self.data[j, i] = 0
            self.k4s = [
                clique for clique in self.k4s
                if (i, j) not in _k4_edge_set(clique)
            ]

    # ── local search ──────────────────────────────────────────────────────────

    def local_search(self, improve_with_local_search):
        self.k4s = _find_k4s(self.data)
        self._remove_edges_greedily()
        if improve_with_local_search:
            self._add_edges_greedily()
        self.k4s = []
        if self.MAKE_OBJECT_CANONICAL:
            self.data = sort_graph_based_on_degree(self.data)
        self.calc_features()
        self.calc_score()

    # ── class-level params (process pool) ────────────────────────────────────

    @classmethod
    def _update_class_params(cls, pars):
        cls.MAKE_OBJECT_CANONICAL, cls.ALPHA_MODE, cls.APPROX_RESTARTS = pars

    @classmethod
    def _save_class_params(cls):
        return (cls.MAKE_OBJECT_CANONICAL, cls.ALPHA_MODE, cls.APPROX_RESTARTS)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class K4FreeEnvironment(BaseEnvironment):
    k = 2
    are_coordinates_symmetric = True
    data_class = K4FreeDataPoint

    def __init__(self, params):
        super().__init__(params)
        self.data_class.MAKE_OBJECT_CANONICAL = params.make_object_canonical
        self.data_class.ALPHA_MODE = params.alpha_mode
        self.data_class.APPROX_RESTARTS = params.approx_restarts

        encoding_augmentation = random_symmetry_adj_matrix if params.augment_data_representation else None

        if params.encoding_tokens == "single_integer":
            self.tokenizer = SparseTokenizerSingleInteger(
                self.data_class, params.N, self.k, self.are_coordinates_symmetric,
                self.SPECIAL_SYMBOLS, encoding_augmentation=encoding_augmentation,
            )
        elif params.encoding_tokens == "sequence_k_tokens":
            self.tokenizer = SparseTokenizerSequenceKTokens(
                self.data_class, params.N, self.k, self.are_coordinates_symmetric,
                self.SPECIAL_SYMBOLS, encoding_augmentation=encoding_augmentation,
            )
        elif params.encoding_tokens == "adjacency":
            self.tokenizer = DenseTokenizer(
                self.data_class, params.N, self.k, self.are_coordinates_symmetric,
                self.SPECIAL_SYMBOLS, pow2base=params.pow2base,
                encoding_augmentation=encoding_augmentation,
            )
        else:
            raise ValueError(f"Invalid encoding_tokens: {params.encoding_tokens}")

    @staticmethod
    def register_args(parser):
        parser.add_argument("--N", type=int, default=10,
                            help="Number of vertices (8–12 for comparison with enumerated graphs)")
        parser.add_argument("--encoding_tokens", type=str, default="single_integer",
                            help="single_integer | sequence_k_tokens | adjacency")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false",
                            help="Sort nodes by degree for canonical deduplication")
        parser.add_argument("--augment_data_representation", type=bool_flag, default="false",
                            help="Augment with random symmetry relabelling during training")
        parser.add_argument("--pow2base", type=int, default=1,
                            help="Bits per token for adjacency encoding")
        parser.add_argument("--alpha_mode", type=str, default="exact",
                            choices=["exact", "approx"],
                            help="'exact': branch-and-bound MIS (use for N<=20). "
                                 "'approx': greedy restarts, optimistic lower bound on alpha.")
        parser.add_argument("--approx_restarts", type=int, default=30,
                            help="Random restarts for approx alpha solver")