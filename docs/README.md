# Cortex Engine - Android JNI Bindings for llama.cpp

**Run Large Language Models on Android devices with full grammar constraint support.**

[![Android](https://img.shields.io/badge/Android-API%2026%2B-green)](https://developer.android.com)
[![Architecture](https://img.shields.io/badge/Arch-ARM64-blue)](https://developer.android.com/ndk/guides/architectures)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🚀 Features

- ✅ **Text Generation** - Chat, completion, FIM (Fill-In-Middle)
- ✅ **Grammar Constraints** - GBNF/JSON Schema for structured output
- ✅ **Embeddings** - Extract vector representations
- ✅ **Reranking** - Document relevance scoring
- ✅ **LoRA Adapters** - Fine-tune models on-device
- ✅ **Session Management** - Save/load conversation state
- ✅ **Production Ready** - No crashes, handles edge cases

## 📱 Requirements

- **Android:** API 26+ (Android 8.0)
- **Architecture:** ARM64 (arm64-v8a)
- **RAM:** 6GB+ recommended (for 7B models)
- **Storage:** 4-8GB per GGUF model

## 🏗️ Architecture

```
Kotlin/Java (UI) 
    ↓ JNI
C++ (JNI Layer)
    ↓ 
llama.cpp (libllama.so)
    ↓
GGML (libggml.so)
```

See [JNI_ARCHITECTURE.md](JNI_ARCHITECTURE.md) for detailed technical documentation.

## 🚀 Quick Start

### 1. Add Dependency

```kotlin
dependencies {
    implementation project(':app')
}
```

### 2. Load Model

```kotlin
val engine = LlamaEngine()
engine.load("/data/local/tmp/model.gguf")
```

### 3. Generate Text

```kotlin
// Simple generation with default parameters
engine.generate("What is AI?").collect { token ->
    print(token)
}

// With custom sampling parameters
val params = SamplingParams(
    temperature = 0.7f,    // Lower = more deterministic
    topK = 50,             // Keep top 50 tokens
    topP = 0.9f,           // Nucleus sampling
    maxTokens = 256        # Limit output length
)
engine.generate("What is AI?", params = params).collect { token ->
    print(token)
}

// Using predefined presets
engine.generate("Creative story", params = SamplingParams.CREATIVE).collect { token ->
    print(token)
}
engine.generate("Factual answer", params = SamplingParams.CONSERVATIVE).collect { token ->
    print(token)
}

// Mirostat v2 for consistent perplexity
engine.generate("Technical documentation", params = SamplingParams.MIROSTAT).collect { token ->
    print(token)
}

// Advanced sampling: Typical-P + XTC
val params = SamplingParams(
    typicalP = 0.95f,        // Filter by entropy
    xtcProbability = 0.5f,   // 50% chance to exclude top choices
    xtcThreshold = 0.1f      // Exclude tokens above 10% probability
)
engine.generate("Poetic verse", params = params).collect { token ->
    print(token)
}

// DRY sampler to reduce repetition
val params = SamplingParams(
    dryMultiplier = 0.5f,    // Enable DRY with medium strength
    dryBase = 1.75f,         // Base penalty
    dryAllowedLength = 2     // Allow 2 char repeats before penalty
)
engine.generate("Story continuation", params = params).collect { token ->
    print(token)
}

// With grammar constraint (valid JSON only)
val grammar = """root ::= "{" content "}" 
content ::= [^}] content |"""
engine.generate("Create JSON profile", grammar).collect { token ->
    print(token)
}

// With stop sequences (early termination)
val params = SamplingParams(
    stopSequences = listOf("<|endoftext|>", "User:", "\n\n")
)
engine.generate("Continue the story", params = params).collect { token ->
    print(token)
}
```

### 4. Cleanup

```kotlin
engine.close()
```

## 📖 API Reference

### Core Methods

| Method | Description |
|--------|-------------|
| `load(path)` | Load GGUF model |
| `generate(prompt, grammar?, params?)` | Stream tokens |
| `infill(prefix, suffix)` | FIM code completion |
| `embedding(text)` | Get embedding vector |
| `rerank(query, docs)` | Score document relevance |
| `tokenize(text)` | Text → token IDs |
| `detokenize(tokens)` | Token IDs → text |
| `saveState(path)` | Save conversation |
| `loadState(path)` | Restore conversation |
| `addLora(path, scale)` | Load LoRA adapter |
| `getModelInfo()` | Get model metadata |
| `getBackendInfo()` | Get device capabilities |
| `getVocabularyInfo()` | Get vocabulary information |
| `resetPerformanceMetrics()` | Reset performance counters |
| `printPerformanceMetrics()` | Print performance report |
| `getContextSize()` | Get context window size |
| `getBatchSize()` | Get batch size |
| `getBench()` | Get performance metrics |
| `memory()` | Get memory control for cache management |
| `logits()` | Get logits sampler for custom sampling |

### Grammar Support

```kotlin
// JSON Schema → GBNF
val schema = """{"type": "object", "properties": {"name": {"type": "string"}}}"""
val grammar = engine.convertJsonSchema(schema)

// Or write GBNF directly
val grammar = """
    root ::= "{" pair "}"
    pair ::= "\"name\"" ":" string
    string ::= "\"" [^"]* "\""
"""
```

### Model Information

```kotlin
// Get model metadata
val info = engine.getModelInfo()
println("Model: ${info.description}")
println("Parameters: ${info.formattedParams}")  // "7B"
println("Size: ${info.formattedSize}")          // "4.2 GB"
println("Layers: ${info.layerCount}")
println("Embedding: ${info.embeddingSize}")
println("Vocab: ${info.vocabSize} tokens")
println("Context: ${info.contextSize}")
println("Type: ${if (info.hasEncoder) "Encoder-Decoder" else "Decoder-only"}")
println("Chat Template: ${info.chatTemplate ?: "None"}")

// Get device capabilities
val backend = engine.getBackendInfo()
println("GPU Offload: ${backend.supportsGpuOffload}")
println("MMAP: ${backend.supportsMmap}")
println("Max Devices: ${backend.maxDevices}")

// Get vocabulary info
val vocab = engine.getVocabularyInfo()
println("Vocab Size: ${vocab.vocabSize}")
println("BOS: ${vocab.bosToken}, EOS: ${vocab.eosToken}")
println("FIM tokens: ${vocab.fimPreToken}, ${vocab.fimMidToken}, ${vocab.fimSufToken}")

// Performance metrics
val metrics = engine.getBench()
println("Tokens/sec: ${metrics.tokensPerSecond}")
println("Total tokens: ${metrics.tokensGenerated}")
engine.printPerformanceMetrics()  // Detailed report to logcat
engine.resetPerformanceMetrics()  // Reset counters
```

### Thread Configuration

```kotlin
// Thread configuration
engine.setThreadConfig(ThreadConfig.PERFORMANCE)  // Use all cores
engine.setThreadConfig(ThreadConfig.BATTERY_SAVER)  // Save battery
engine.setThreadConfig(ThreadConfig(nThreads = 4))  // Custom config

// Get device info
val cores = engine.getCpuCoreCount()

// Check current configuration
val config = engine.getThreadConfig()
println("Using ${config.effectiveThreads()} threads")
```

### Model Metadata

```kotlin
// Get all metadata
val metadata = engine.getModelMetadata()
println("Architecture: ${metadata.architecture}")
println("Quantization: ${metadata.quantization}")
println("Author: ${metadata.author}")
println("License: ${metadata.license}")

// Get specific value
val arch = engine.getMetadataValue("general.architecture")
println("Model type: $arch")

// List all metadata keys
val count = engine.getMetadataCount()
println("Total metadata entries: $count")

// Get available chat templates
val templates = engine.getBuiltinTemplates()
println("Available templates: $templates")

// Get full metadata summary
println(metadata.getSummary())
```

### Memory Management

```kotlin
// Get memory control instance
val memory = engine.memory()

// Clear cache to free memory
memory.clearCache()

// Remove old tokens (sliding window)
memory.removeTokens(pos0 = 0, pos1 = 1000)

// Copy sequence for branching conversations
memory.copySequence(srcSeqId = 0, dstSeqId = 1)

// Keep only specific sequence
memory.keepOnlySequence(seqId = 0)

// Position interpolation for long context
memory.dividePositions(pos0 = 0, pos1 = -1, divisor = 8)

// Get cache statistics
val stats = memory.getStats()
println("Cache usage: ${stats.usagePercent()}%")
```

### LoRA Adapters

```kotlin
// Load multiple LoRAs
engine.addLora("/path/to/lora1.gguf", scale = 0.8f)
engine.addLora("/path/to/lora2.gguf", scale = 0.5f)

// Check loaded LoRAs
println("Loaded ${engine.getLoraCount()} LoRAs")
engine.getLoadedLoras().forEach { println("  - $it") }

// Remove specific LoRA
engine.removeLora("/path/to/lora1.gguf")

// Clear all
engine.resetLoras()
```

### Session Management

```kotlin
// Save different conversation states
engine.saveState("/path/conv1.bin", seqId = 0)
engine.saveState("/path/conv2.bin", seqId = 1)

// Load specific sequence
engine.loadState("/path/conv1.bin", seqId = 0)
```

### Logits Access (Custom Sampling)

```kotlin
// Get logits sampler
val logits = engine.logits()

// Get raw logits for the last token
val logitsArray = logits.getLogits()

// Custom sampling - greedy decoding
val tokenId = logits.argmax(logitsArray!!)

// Custom sampling - softmax with temperature
val tokenId = logits.sampleSoftmax(logitsArray, temperature = 0.8f)

// Get top 10 most likely tokens
val top10 = logits.topK(logitsArray, k = 10)
top10.forEach { (tokenId, prob) ->
    println("Token $tokenId: ${prob * 100}%")
}

// Apply custom filters
var filtered = logits.applyTemperature(logitsArray, 0.7f)
filtered = logits.applyTopP(filtered, 0.9f)
filtered = logits.applyRepetitionPenalty(filtered, previousTokens, 1.2f)

// Full custom generation with callback
logits.generateWithCustomSampler(
    prompt = "Hello",
    maxTokens = 100
) { logits, position ->
    // Your custom sampling logic here
    logits.argmax(logits)
}.collect { token ->
    print(token)
}
```

## 🔧 Building

### Prerequisites
- Android Studio
- NDK r25 or newer
- CMake 3.22+

### Build Commands

```bash
# Debug build
./gradlew assembleDebug

# Release build
./gradlew assembleRelease
```

Output: `app/build/outputs/apk/debug/app-debug.apk`

## 🐛 Troubleshooting

### Common Issues

**1. Model fails to load**
```
Check: File exists, has read permissions, is valid GGUF format
```

**2. Out of memory**
```
Solution: Use quantized model (Q4_K_M), reduce context size
```

**3. Grammar not working**
```
Check: Grammar syntax valid, PCRE2 linked properly
```

See [TROUBLESHOOTING_LOG.md](TROUBLESHOOTING_LOG.md) for detailed solutions.

## 📊 Performance

| Device | Model | Speed |
|--------|-------|-------|
| Pixel 7 | 7B Q4 | ~5 t/s |
| Galaxy S23 | 7B Q4 | ~8 t/s |
| OnePlus 12 | 7B Q4 | ~10 t/s |

*Note: CPU only. GPU via Vulkan disabled in current build.*

## 🏛️ Project Structure

```
├── app/src/main/
│   ├── cpp/jni/          # JNI C++ implementation
│   ├── java/             # Kotlin API
│   └── assets/           # Model files
├── docs/                 # Documentation
│   ├── JNI_ARCHITECTURE.md
│   └── TROUBLESHOOTING_LOG.md
└── CMakeLists.txt        # Native build config
```

## 🤝 Contributing

Contributions welcome! Please:
1. Test on actual Android devices
2. Follow existing code style
3. Update documentation
4. Add tests for new features

## 📜 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Core inference engine
- [GGML](https://github.com/ggerganov/ggml) - Tensor operations
- [PCRE2](https://www.pcre.org/) - Safe regex for Android

---

**Made with ❤️ for on-device AI**
