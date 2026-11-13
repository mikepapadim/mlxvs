# Copyright © 2026 mikepapadim

"""cuVS's Python API on Apple silicon: ``import mlxvs.cuvs as cuvs``.

The modules, classes, functions, parameters and results of NVIDIA cuVS
(26.10), on mlxvs's Metal kernels, so code written for cuVS runs on a Mac by
changing its imports::

    from mlxvs.cuvs.neighbors import cagra       # from cuvs.neighbors import cagra
    from mlxvs.cuvs.common import device_ndarray  # from pylibraft.common import device_ndarray

    index = cagra.build(cagra.IndexParams(graph_degree=32), dataset)
    distances, neighbors = cagra.search(cagra.SearchParams(), index, queries, k=10)

CAGRA and brute-force index files are cuVS's own formats: an index built and
saved with cuVS on an NVIDIA GPU loads here, and back.
"""

from mlxvs.cuvs import cluster, common, distance, neighbors

__all__ = ["cluster", "common", "distance", "neighbors"]
