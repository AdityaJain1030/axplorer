"""
K4-free graph environment for Axplorer.

Goal: find K4-free graphs on N vertices that maximise N - alpha(G),
where alpha(G) is the independence number.

Score = N - alpha  (higher is better).
score = -1 is reserved for INVALID (K4-containing) graphs.

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

def _build_nbr_masks(adj: np.ndarray):
    """Build bitmask neighbor lists from adjacency matrix."""
    N = adj.shape[0]
    nbr = [0] * N
    for i in range(N):
        for j in np.nonzero(adj[i])[0]:
            nbr[i] |= 1 << int(j)
    return nbr


def _alpha_exact(nbr, N: int) -> int:
    """
    Branch-and-bound exact independence number with degree-based
    vertex ordering for tighter pruning.
    """
    deg = [bin(nbr[v]).count('1') for v in range(N)]
    order = sorted(range(N), key=lambda v: -deg[v])
    rank_of = [0] * N
    for r, v in enumerate(order):
        rank_of[v] = r

    rnbr = [0] * N
    for r in range(N):
        v = order[r]
        m = nbr[v]
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            rnbr[r] |= 1 << rank_of[u]
            m ^= lsb

    best = [0]

    def bb(candidates: int, size: int):
        if candidates == 0:
            if size > best[0]:
                best[0] = size
            return
        remaining = bin(candidates).count('1')
        if size + remaining <= best[0]:
            return
        v = (candidates & -candidates).bit_length() - 1
        bb(candidates & ~rnbr[v] & ~(1 << v), size + 1)
        bb(candidates & ~(1 << v), size)

    bb((1 << N) - 1, 0)
    return best[0]


def _alpha_approx(nbr, N: int, n_restarts: int = 30) -> int:
    """
    Greedy independent set with mixed restart strategy.
    Even-numbered restarts use min-remaining-degree heuristic for quality.
    Odd-numbered restarts use pure random order for diversity.
    Returns the largest IS found (lower bound on alpha).
    """
    best = 0
    vertices = list(range(N))

    for restart in range(n_restarts):
        random.shuffle(vertices)
        use_min_degree = (restart % 2 == 0)

        if use_min_degree:
            # ── min-degree greedy with shuffle-based tie-breaking ─────
            priority = [0] * N
            for rank, v in enumerate(vertices):
                priority[v] = rank

            available = (1 << N) - 1
            # Maintain available-degree per vertex incrementally
            avail_deg = [0] * N
            tmp = available
            while tmp:
                lsb = tmp & -tmp
                v = lsb.bit_length() - 1
                tmp ^= lsb
                avail_deg[v] = bin(nbr[v] & available).count('1')

            size = 0
            while available:
                best_v = -1
                best_d = N + 1
                best_p = N + 1
                tmp = available
                while tmp:
                    lsb = tmp & -tmp
                    v = lsb.bit_length() - 1
                    tmp ^= lsb
                    d = avail_deg[v]
                    if d < best_d or (d == best_d and priority[v] < best_p):
                        best_d = d
                        best_v = v
                        best_p = priority[v]

                size += 1
                # Remove best_v and its neighbours from available
                removed = available & (nbr[best_v] | (1 << best_v))
                available &= ~removed
                # Decrement avail_deg for vertices still available that
                # were adjacent to any removed vertex
                r = removed
                while r:
                    lsb = r & -r
                    u = lsb.bit_length() - 1
                    r ^= lsb
                    affected = nbr[u] & available
                    while affected:
                        alb = affected & -affected
                        w = alb.bit_length() - 1
                        affected ^= alb
                        avail_deg[w] -= 1
        else:
            # ── pure random greedy ────────────────────────────────────
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

def _adding_edge_creates_k4(nbr, u: int, v: int) -> bool:
    """
    Fast check: does adding edge (u,v) create a K4?
    A K4 through (u,v) requires two common neighbours c,d of u and v
    (before the new edge) such that c-d is also an edge.
    """
    common = nbr[u] & nbr[v]
    if bin(common).count('1') < 2:
        return False
    tmp = common
    while tmp:
        lsb = tmp & -tmp
        c = lsb.bit_length() - 1
        tmp ^= lsb
        if nbr[c] & (common ^ lsb):
            return True
    return False


def _find_k4s(nbr, N: int):
    """Return list of all K4s as sorted 4-tuples of vertex indices."""
    k4s = []
    for a in range(N):
        na = nbr[a]
        for b in range(a + 1, N):
            if not (na >> b & 1):
                continue
            c_bits = na & nbr[b] & ~((1 << (b + 1)) - 1)
            while c_bits:
                lsb_c = c_bits & -c_bits
                c = lsb_c.bit_length() - 1
                c_bits ^= lsb_c
                d_bits = na & nbr[b] & nbr[c] & ~((1 << (c + 1)) - 1)
                while d_bits:
                    lsb_d = d_bits & -d_bits
                    d = lsb_d.bit_length() - 1
                    d_bits ^= lsb_d
                    k4s.append((a, b, c, d))
    return k4s


def _k4_edge_set(clique):
    a, b, c, d = clique
    return frozenset(((a,b),(a,c),(a,d),(b,c),(b,d),(c,d)))


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

    def _compute_alpha(self, nbr) -> int:
        if self.ALPHA_MODE == "exact":
            return _alpha_exact(nbr, self.N)
        return _alpha_approx(nbr, self.N, n_restarts=self.APPROX_RESTARTS)

    def calc_score(self, nbr=None):
        if self.k4s:
            self.score = -1
            return

        if nbr is None:
            nbr = _build_nbr_masks(self.data)
        alpha = self._compute_alpha(nbr)

        if alpha == 0:
            self.score = 0
            return

        self.score = self.N - alpha

    def calc_features(self):
        w = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                w.append(self.data[i, j])
        self.features = ",".join(map(str, w))

    # ── greedy construction ───────────────────────────────────────────────────

    def _add_edges_greedily(self, nbr=None):
        """
        Add edges in random order, skipping any that would create a K4.
        Maintains nbr masks incrementally. Caller may supply existing nbr
        to avoid a redundant rebuild.
        """
        N = self.N
        if nbr is None:
            nbr = _build_nbr_masks(self.data)

        candidates = [
            (i, j) for i in range(N) for j in range(i + 1, N)
            if self.data[i, j] == 0
        ]
        random.shuffle(candidates)

        for i, j in candidates:
            if self.data[i, j] == 0:
                if _adding_edge_creates_k4(nbr, i, j):
                    continue
                self.data[i, j] = 1
                self.data[j, i] = 1
                nbr[i] |= 1 << j
                nbr[j] |= 1 << i

        return nbr

    # ── K4 removal ────────────────────────────────────────────────────────────

    def _remove_edges_greedily(self, nbr):
        """
        Remove the edge participating in the most K4s, repeat until K4-free.
        Recomputes K4 list from scratch after each removal to avoid stale
        entries from edge-set overlap between cliques.
        Returns the (mutated) nbr masks.
        """
        while True:
            self.k4s = _find_k4s(nbr, self.N)
            if not self.k4s:
                break

            clique_edges = [_k4_edge_set(c) for c in self.k4s]
            edge_count = {}
            for es in clique_edges:
                for e in es:
                    edge_count[e] = edge_count.get(e, 0) + 1

            i, j = max(edge_count, key=edge_count.get)
            self.data[i, j] = 0
            self.data[j, i] = 0
            nbr[i] &= ~(1 << j)
            nbr[j] &= ~(1 << i)

        return nbr

    # ── local search ──────────────────────────────────────────────────────────

    def local_search(self, improve_with_local_search):
        nbr = _build_nbr_masks(self.data)
        nbr = self._remove_edges_greedily(nbr)
        if improve_with_local_search:
            nbr = self._add_edges_greedily(nbr)
        self.k4s = []
        if self.MAKE_OBJECT_CANONICAL:
            self.data = sort_graph_based_on_degree(self.data)
            nbr = _build_nbr_masks(self.data)
        self.calc_features()
        self.calc_score(nbr=nbr)

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