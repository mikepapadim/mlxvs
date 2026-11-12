// Copyright © 2026 mikepapadim

import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Searches an index built on the GPU by mlxvs and saved with CagraIndex.save_jvector.
 *
 * <p>Arguments: the index file; the queries as raw little-endian float32 rows (numpy's
 * {@code queries.astype("<f4").tofile(path)}); k; rerankK (JVector's search list length, as
 * mlxvs's itopk_size); the similarity (euclidean, cosine or dot_product); the file the ids are
 * written to (little-endian int32, nq x k, -1 where fewer were found).
 *
 * <pre>
 * mvn -q compile exec:exec -Dexec.args="--add-modules jdk.incubator.vector -classpath %classpath \
 *     Search index.jvector queries.f32 10 64 euclidean ids.i32"
 * </pre>
 */
public class Search {
  public static void main(String[] args) throws Exception {
    if (args.length != 6) {
      System.err.println("Search <index> <queries.f32> <k> <rerankK> <similarity> <ids.i32>");
      System.exit(2);
    }
    var index = OnDiskGraphIndex.load(ReaderSupplierFactory.open(Path.of(args[0])));
    int k = Integer.parseInt(args[2]);
    int rerankK = Integer.parseInt(args[3]);
    var similarity = VectorSimilarityFunction.valueOf(args[4].toUpperCase());
    int d = index.getDimension();
    var queries =
        ByteBuffer.wrap(Files.readAllBytes(Path.of(args[1])))
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer();
    int nq = queries.limit() / d;
    System.out.printf("%d nodes of degree %d, %d dimensions; %d queries%n",
        index.size(0), index.maxDegree(), d, nq);

    var vectors = VectorizationProvider.getInstance().getVectorTypeSupport();
    var ids = ByteBuffer.allocate(nq * k * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    long start = System.nanoTime();
    try (var view = index.getView();
        var searcher = new GraphSearcher(index)) {
      for (int i = 0; i < nq; i++) {
        float[] q = new float[d];
        queries.get(i * d, q);
        // exact scores from the vectors stored inline in the index
        var scores = DefaultSearchScoreProvider.exact(vectors.createFloatVector(q), similarity, view);
        var nodes = searcher.search(scores, k, rerankK, 0f, 0f, Bits.ALL).getNodes();
        for (int j = 0; j < k; j++) {
          ids.putInt(j < nodes.length ? nodes[j].node : -1);
        }
      }
    }
    System.out.printf("%.3f ms per query on one thread%n", (System.nanoTime() - start) / 1e6 / nq);
    Files.write(Path.of(args[5]), ids.array());
  }
}
