# Build on the GPU, search with JVector

`CagraIndex.save_jvector` writes the index in [JVector](https://github.com/datastax/jvector)'s on-disk format
(`OnDiskGraphIndex`, format version 6 as JVector 4.0 writes it): the CAGRA graph as one layer entered at the
medoid, and the rows as inline float32 vectors. `Search.java` loads such a file with JVector 4.0.2 and searches it.

```python
import mlxvs, numpy as np

index = mlxvs.CagraIndex.build(vectors, graph_degree=32, metric="l2")
index.save_jvector("index.jvector")
queries.astype("<f4").tofile("queries.f32")
```

```bash
cd examples/jvector
mvn -q compile exec:exec -Dexec.args="--add-modules jdk.incubator.vector -classpath %classpath \
    Search index.jvector queries.f32 10 64 euclidean ids.i32"
```

```python
ids = np.fromfile("ids.i32", dtype="<i4").reshape(-1, 10)
```

On SIFT1M (graph degree 32 built in about 10 s), JVector finds recall@10 0.978 with `rerankK` 64 and 0.994 with
128 (0.5 and 0.8 ms per query on one core of an M3 Pro). Use `cosine` for an index built with `"cosine"` (its rows
are stored normalized) and `dot_product` for `"inner_product"`.

From Java alone: build the index on the GPU through mlxvs's C API (`mlxvs_index_build`, then
`mlxvs_index_save_jvector`), as [examples/java_ffm](../java_ffm/MlxvsExample.java) does, and search the file with
JVector as above.
