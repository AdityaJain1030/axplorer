"""
KFour environment for Axplorer.

Complement-graph formulation of the K₄-free independence conjecture.

We build H (the complement of G) on N vertices and maximize |E(H)| subject to:
  1. H is K_t-free          (no clique of size t in H)
              ↔ alpha(G) ≤ t-1  in the original graph
  2. alpha(H) ≤ 3           (no independent set of size 4 in H)
              ↔ G is K₄-free

Score = |E(H)| if both constraints hold, -1 otherwise.

Higher score → fewer edges in G → smaller d_avg(G) → tighter lower bound on c:
    c_lb = (t-1) * d_avg / (N * ln(d_avg))
"""

import numpy as np

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import DenseTokenizer, SparseTokenizerSequenceKTokens, SparseTokenizerSingleInteger
from src.envs.utils import random_symmetry_adj_matrix, sort_graph_based_on_degree
from src.utils import bool_flag


# ─────────────────────────────────────────────────────────────────────────────
# Bitmask helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_nbr_masks(adj: np.ndarray):
    """Build integer bitmask neighbor list from adjacency matrix."""
    N = adj.shape[0]
    nbr = [0] * N
    for i in range(N):
        for j in np.nonzero(adj[i])[0]:
            nbr[i] |= 1 << int(j)
    return nbr


def _has_clique(size: int, candidates: int, nbr: list) -> bool:
    """
    Return True iff the subgraph induced by `candidates` (a bitmask) contains
    a clique of `size` vertices, using the provided neighbor masks.

    Enumerates vertices in increasing order to avoid double-counting.
    Edge cases:
      size=0  → True  (vacuously; used for t=2: any edge creates K_2)
      size=1  → True iff candidates is non-empty
    """
    if size == 0:
        return True
    tmp = candidates
    while tmp:
        lsb = tmp & -tmp
        v = lsb.bit_length() - 1
        tmp ^= lsb
        # Include v; look for rest among neighbors of v that are strictly after v
        if _has_clique(size - 1, nbr[v] & tmp, nbr):
            return True
    return False


def _find_all_kt(t: int, nbr: list, N: int) -> list:
    """Return all K_t cliques as sorted t-tuples of vertex indices."""
    result = []

    def collect(size, path, candidates):
        if size == 0:
            result.append(tuple(path))
            return
        tmp = candidates
        while tmp:
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            tmp ^= lsb
            collect(size - 1, path + [v], nbr[v] & tmp)

    collect(t, [], (1 << N) - 1)
    return result


def _find_fixable_i4(comp_nbr: list, N: int, nbr: list, t: int):
    """
    Search for an I_4 in H (4-clique in complement) where at least one of the
    six vertex pairs can be connected in H without creating K_t.

    Returns (i4_tuple, (x, y)) where (x,y) is the K_t-safe edge to add,
    or None if every I_4 is fully blocked.

    All six pairs in an I_4 are non-edges in H by definition, so no existence
    check on self.data is needed.
    """
    def search(size, path, candidates):
        if size == 0:
            a, b, c, d = path
            for x, y in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
                # Adding (x,y) creates K_t iff common neighbourhood has K_{t-2}
                if not _has_clique(t - 2, nbr[x] & nbr[y], nbr):
                    return (tuple(path), (x, y))
            return None  # this I_4 is fully blocked
        tmp = candidates
        while tmp:
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            tmp ^= lsb
            result = search(size - 1, path + [v], comp_nbr[v] & tmp)
            if result is not None:
                return result
        return None

    return search(4, [], (1 << N) - 1)


def _find_swap_move(comp_nbr: list, N: int, nbr: list, t: int):
    """
    When every I_4 pair is blocked (adding it would create K_t), look for a
    swap: remove one edge (ri, rj) from H so that some I_4 pair (ax, ay)
    becomes K_t-safe to add.

    The swap is valid iff after removing (ri,rj), adding (ax,ay) does not
    create K_t.  Both operations are tested atomically (nbr is restored).

    Returns ((ri, rj), (ax, ay)) or None if no such swap exists.
    """
    def search(size, path, candidates):
        if size == 0:
            a, b, c, d = path
            for x, y in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
                common = nbr[x] & nbr[y]
                if not _has_clique(t - 2, common, nbr):
                    continue  # already fixable without swap
                # Try removing each edge (u,v) that lies inside `common`
                tmp_u = common
                while tmp_u:
                    lsb_u = tmp_u & -tmp_u
                    u = lsb_u.bit_length() - 1
                    tmp_u ^= lsb_u
                    # v must be > u, also in common, and adjacent to u in H
                    tmp_v = nbr[u] & common & ~((1 << (u + 1)) - 1)
                    while tmp_v:
                        lsb_v = tmp_v & -tmp_v
                        v = lsb_v.bit_length() - 1
                        tmp_v ^= lsb_v
                        # Tentatively remove (u, v) from nbr
                        nbr[u] &= ~(1 << v)
                        nbr[v] &= ~(1 << u)
                        safe = not _has_clique(t - 2, nbr[x] & nbr[y], nbr)
                        nbr[u] |= 1 << v   # restore
                        nbr[v] |= 1 << u
                        if safe:
                            return ((u, v), (x, y))
            return None
        tmp = candidates
        while tmp:
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            tmp ^= lsb
            result = search(size - 1, path + [v], comp_nbr[v] & tmp)
            if result is not None:
                return result
        return None

    return search(4, [], (1 << N) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# DataPoint
# ─────────────────────────────────────────────────────────────────────────────

class KFourDataPoint(DataPoint):
    MAKE_OBJECT_CANONICAL = False
    T = 4  # K_t-free clique bound; set at class level for multiprocessing

    def __init__(self, N, t=None, init=False):
        super().__init__()
        self.N = N
        self.t = self.__class__.T if t is None else t
        self.data = np.zeros((N, N), dtype=np.uint8)

        if init:
            self._add_edges_greedily()
            if self.MAKE_OBJECT_CANONICAL:
                self.data = sort_graph_based_on_degree(self.data)
            self.calc_features()
            self.calc_score()

    # ── scoring ───────────────────────────────────────────────────────────────

    def calc_score(self):
        N = self.N
        nbr = _build_nbr_masks(self.data)
        all_mask = (1 << N) - 1

        # Constraint 1: H must be K_t-free
        if _has_clique(self.t, all_mask, nbr):
            self.score = -1
            return -1

        # Constraint 2: alpha(H) <= 3  (no I_4 in H = no K_4 in complement of H)
        comp_nbr = [(all_mask ^ nbr[v]) & ~(1 << v) for v in range(N)]
        if _has_clique(4, all_mask, comp_nbr):
            self.score = -1
            return -1

        self.score = int(self.data.sum()) // 2
        return self.score

    def calc_features(self):
        w = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                w.append(self.data[i, j])
        self.features = ",".join(map(str, w))

    def local_search(self, improve_with_local_search):
        N = self.N
        all_mask = (1 << N) - 1

        # Each restart generates a fresh random K_t-free graph (via
        # _add_edges_greedily), which independently has a ~3-5% chance of
        # satisfying alpha(H) ≤ 3.  200 restarts gives >99.9% success
        # probability for N < R(4,t).  Phase 2 direct additions are kept as a
        # cheap first attempt before giving up on the current graph.
        repaired = False
        for _ in range(200):
            nbr = _build_nbr_masks(self.data)

            # ── Phase 1: repair K_t violations ───────────────────────────────
            while _has_clique(self.t, all_mask, nbr):
                cliques = _find_all_kt(self.t, nbr, N)
                edge_count = {}
                for clique in cliques:
                    for a in range(len(clique)):
                        for b in range(a + 1, len(clique)):
                            e = (clique[a], clique[b])
                            edge_count[e] = edge_count.get(e, 0) + 1
                i, j = max(edge_count, key=edge_count.get)
                self.data[i, j] = 0
                self.data[j, i] = 0
                nbr[i] &= ~(1 << j)
                nbr[j] &= ~(1 << i)

            # ── Phase 2: repair I_4 violations by direct safe additions ──────
            comp_nbr = [(all_mask ^ nbr[v]) & ~(1 << v) for v in range(N)]
            if not _has_clique(4, all_mask, comp_nbr):
                repaired = True
                break
            while True:
                hit = _find_fixable_i4(comp_nbr, N, nbr, self.t)
                if hit is None:
                    break
                _, (x, y) = hit
                self.data[x, y] = 1
                self.data[y, x] = 1
                nbr[x] |= 1 << y
                nbr[y] |= 1 << x
                comp_nbr = [(all_mask ^ nbr[v]) & ~(1 << v) for v in range(N)]
                if not _has_clique(4, all_mask, comp_nbr):
                    repaired = True
                    break

            if repaired:
                break

            # Restart: fresh random K_t-free graph, retry both phases.
            self.data = np.zeros((N, N), dtype=np.uint8)
            self._add_edges_greedily()

        # ── Phase 3: greedy densify (only when improve_with_local_search) ────
        # Adding edges can only reduce independent sets (alpha is non-increasing
        # under edge addition), so the I_4 constraint is maintained for free.
        # We only need to guard against K_t.
        if improve_with_local_search:
            nbr = _build_nbr_masks(self.data)
            candidates = [
                (i, j) for i in range(N) for j in range(i + 1, N)
                if self.data[i, j] == 0
            ]
            np.random.shuffle(candidates)
            for i, j in candidates:
                common = nbr[i] & nbr[j]
                if not _has_clique(self.t - 2, common, nbr):
                    self.data[i, j] = 1
                    self.data[j, i] = 1
                    nbr[i] |= 1 << j
                    nbr[j] |= 1 << i

        if self.MAKE_OBJECT_CANONICAL:
            self.data = sort_graph_based_on_degree(self.data)
        self.calc_features()
        self.calc_score()

    # ── greedy construction ───────────────────────────────────────────────────

    def _add_edges_greedily(self):
        """
        Add edges to H in random order while keeping H K_t-free.
        Does not enforce alpha(H) <= 3 during construction; calc_score will
        catch any I_4 violations (some init samples may score -1).
        """
        t = self.t
        N = self.N
        nbr = [0] * N

        candidates = [(i, j) for i in range(N) for j in range(i + 1, N)]
        np.random.shuffle(candidates)

        for i, j in candidates:
            # Adding edge (i,j) creates K_t iff common neighborhood contains K_{t-2}
            common = nbr[i] & nbr[j]
            if _has_clique(t - 2, common, nbr):
                continue
            # Accept: add edge to H
            self.data[i, j] = 1
            self.data[j, i] = 1
            nbr[i] |= 1 << j
            nbr[j] |= 1 << i

    # ── class-level params (process pool) ────────────────────────────────────

    @classmethod
    def _update_class_params(cls, pars):
        cls.MAKE_OBJECT_CANONICAL, cls.T = pars

    @classmethod
    def _save_class_params(cls):
        return (cls.MAKE_OBJECT_CANONICAL, cls.T)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class KFourEnvironment(BaseEnvironment):
    k = 2
    are_coordinates_symmetric = True
    data_class = KFourDataPoint

    def __init__(self, params):
        super().__init__(params)
        self.data_class.T = params.t
        self.data_class.MAKE_OBJECT_CANONICAL = params.make_object_canonical

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
                            help="Number of vertices in H")
        parser.add_argument("--t", type=int, default=4,
                            help="Clique bound: H must be K_t-free")
        parser.add_argument("--encoding_tokens", type=str, default="single_integer",
                            help="single_integer | sequence_k_tokens | adjacency")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false",
                            help="Sort nodes by degree for canonical deduplication")
        parser.add_argument("--augment_data_representation", type=bool_flag, default="false",
                            help="Augment with random symmetry relabelling during training")
        parser.add_argument("--pow2base", type=int, default=1,
                            help="Bits per token for adjacency encoding")
