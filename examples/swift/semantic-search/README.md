# semantic-search (Swift)

Embeds the paragraphs of a folder's `.md` and `.txt` files with an embedding model on the GPU
([mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)'s `MLXEmbedders`, `BAAI/bge-small-en-v1.5`), then answers
queries with the nearest paragraphs, searched on the GPU with `MLXVS`: exactly (`bruteForceKNN`) for up to 2,000
paragraphs, from a CAGRA graph index built on the device (`CagraIndex.build`) beyond.

```
swift run semantic-search ../../.. "how do I search on an iPhone" "compress the index to save memory"
```

```
embedded 472 paragraphs in 4.4 s
searched in 1.3 ms (exact)

how do I search on an iPhone
  0.644  README.md: let index = try CagraIndex.load(url: Bundle.main.url(forResource: "docs", withExtension: "safetensors")!) let ...
```

The model downloads from the Hugging Face Hub on the first run. Needs macOS 14+ on Apple silicon and Xcode 26+ with
its Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`).
