// Copyright © 2026 mikepapadim

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.nio.file.Path;
import java.util.Random;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * mlxvs from Java through the Foreign Function and Memory API (Java 22+), no JNI: builds a GPU index
 * of random rows, searches it, saves and loads it, and writes it in JVector's format.
 *
 * <pre>
 * cmake -S . -B build && cmake --build build      # builds build/libmlxvs_c.dylib
 * java --enable-native-access=ALL-UNNAMED examples/java_ffm/MlxvsExample.java build/libmlxvs_c.dylib
 * </pre>
 */
public class MlxvsExample {
  static final int L2 = 1;

  /** The C API's functions, bound once. */
  static final class Mlxvs {
    final MethodHandle build, search, save, saveJVector, load, free, lastError, version,
        searchParamsInit;
    final StructLayout searchParams =
        MemoryLayout.structLayout(
            JAVA_INT.withName("itopk_size"),
            JAVA_INT.withName("search_width"),
            JAVA_INT.withName("algo"),
            JAVA_INT.withName("max_iterations"),
            JAVA_INT.withName("min_iterations"),
            JAVA_INT.withName("team_size"),
            JAVA_INT.withName("num_random_samplings"),
            MemoryLayout.paddingLayout(4),
            JAVA_LONG.withName("seed"));
    final VarHandle itopk =
        searchParams.varHandle(MemoryLayout.PathElement.groupElement("itopk_size"));

    Mlxvs(Path library, Arena arena) {
      var lib = SymbolLookup.libraryLookup(library, arena);
      var linker = Linker.nativeLinker();
      java.util.function.BiFunction<String, FunctionDescriptor, MethodHandle> bind =
          (name, fd) -> linker.downcallHandle(lib.find(name).orElseThrow(), fd);
      build = bind.apply("mlxvs_index_build",
          FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS));
      search = bind.apply("mlxvs_index_search",
          FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
      save = bind.apply("mlxvs_index_save", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
      saveJVector =
          bind.apply("mlxvs_index_save_jvector", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
      load = bind.apply("mlxvs_index_load", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
      free = bind.apply("mlxvs_index_free", FunctionDescriptor.ofVoid(ADDRESS));
      lastError = bind.apply("mlxvs_last_error", FunctionDescriptor.of(ADDRESS));
      version = bind.apply("mlxvs_version", FunctionDescriptor.of(ADDRESS));
      searchParamsInit = bind.apply("mlxvs_search_params_init", FunctionDescriptor.ofVoid(ADDRESS));
    }

    void ok(int status) throws Throwable {
      if (status != 0) {
        var msg = ((MemorySegment) lastError.invokeExact()).reinterpret(4096).getString(0);
        throw new IllegalStateException("mlxvs error " + status + ": " + msg);
      }
    }
  }

  public static void main(String[] args) throws Throwable {
    int n = 200_000, d = 64, nq = 1000, k = 10;
    var rng = new Random(42);
    float[] rows = new float[n * d];
    for (int i = 0; i < rows.length; i++) rows[i] = (float) rng.nextGaussian();
    float[] queries = new float[nq * d];
    System.arraycopy(rows, 0, queries, 0, nq * d); // the first rows: each its own nearest

    try (var arena = Arena.ofConfined()) {
      var mlxvs = new Mlxvs(Path.of(args.length > 0 ? args[0] : "build/libmlxvs_c.dylib"), arena);
      var version = ((MemorySegment) mlxvs.version.invokeExact()).reinterpret(64).getString(0);
      System.out.println("mlxvs " + version);

      var rowsSeg = arena.allocateFrom(JAVA_FLOAT, rows);
      var indexOut = arena.allocate(ADDRESS);
      long t0 = System.nanoTime();
      mlxvs.ok((int) mlxvs.build.invokeExact(
          rowsSeg, (long) n, d, L2, MemorySegment.NULL, indexOut));
      var index = indexOut.get(ADDRESS, 0);
      System.out.printf("built %d x %d in %.2f s%n", n, d, (System.nanoTime() - t0) / 1e9);

      var params = arena.allocate(mlxvs.searchParams);
      mlxvs.searchParamsInit.invokeExact(params);
      mlxvs.itopk.set(params, 0L, 128);
      var querySeg = arena.allocateFrom(JAVA_FLOAT, queries);
      var ids = arena.allocate(JAVA_INT, (long) nq * k);
      var dist = arena.allocate(JAVA_FLOAT, (long) nq * k);
      mlxvs.ok((int) mlxvs.search.invokeExact(index, querySeg, (long) nq, k, params, ids, dist));
      t0 = System.nanoTime();
      mlxvs.ok((int) mlxvs.search.invokeExact(index, querySeg, (long) nq, k, params, ids, dist));
      double sec = (System.nanoTime() - t0) / 1e9;
      int self = 0;
      for (int i = 0; i < nq; i++) self += ids.getAtIndex(JAVA_INT, (long) i * k) == i ? 1 : 0;
      System.out.printf("%d queries in %.2f ms (%.0f queries/s); %d found themselves first%n",
          nq, sec * 1e3, nq / sec, self);

      var path = arena.allocateFrom("/tmp/mlxvs_java.safetensors");
      mlxvs.ok((int) mlxvs.save.invokeExact(index, path));
      mlxvs.free.invokeExact(index);
      mlxvs.ok((int) mlxvs.load.invokeExact(path, indexOut));
      index = indexOut.get(ADDRESS, 0);
      mlxvs.ok((int) mlxvs.search.invokeExact(index, querySeg, 1L, k, MemorySegment.NULL, ids, dist));
      System.out.println("loaded; query 0's nearest: " + ids.getAtIndex(JAVA_INT, 0));

      // the same index for JVector (examples/jvector searches it)
      var jvector = arena.allocateFrom("/tmp/mlxvs_java.jvector");
      mlxvs.ok((int) mlxvs.saveJVector.invokeExact(index, jvector));
      System.out.println("wrote /tmp/mlxvs_java.jvector for JVector");
      mlxvs.free.invokeExact(index);
    }
  }
}
