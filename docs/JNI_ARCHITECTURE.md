# JNI Architecture & Implementation Guide

**Project:** Cortex Engine - Android JNI Bindings for llama.cpp  
**Last Updated:** Feb 2, 2026  
**Status:** ✅ Production Ready

## Overview

This document describes the complete JNI (Java Native Interface) architecture that bridges the Kotlin/Java Android layer with the llama.cpp C++ library. The bindings enable on-device LLM inference, embeddings, reranking, and grammar-constrained generation on Android devices.

---

## Architecture Layers

```
┌─────────────────────────────────────────┐
│  Kotlin/Java Layer (MainActivity.kt)    │
│  - LlamaEngine.kt (API wrapper)         │
│  - UI & Business Logic                  │
├─────────────────────────────────────────┤
│  JNI Layer (C++)                        │
│  - jni_*.cpp files                      │
│  - JNI method implementations           │
├─────────────────────────────────────────┤
│  llama.cpp Core                         │
│  - libllama.so                          │
│  - libggml.so                           │
│  - Grammar parsing (PCRE2)              │
└─────────────────────────────────────────┘
```

---

## File Structure

```
app/src/main/cpp/
├── jni/
│   ├── jni_common.cpp/h       # Common utilities, logging, JVM management
│   ├── jni_lifecycle.cpp      # Model loading/unloading
│   ├── jni_completion.cpp     # Text generation (chat, FIM)
│   ├── jni_embedding.cpp      # Embedding extraction
│   ├── jni_grammar.cpp        # GBNF grammar support
│   ├── jni_lora.cpp           # LoRA adapter support
│   ├── jni_rerank.cpp         # Document reranking
│   ├── jni_session.cpp        # Session save/load
│   └── jni_utils.cpp          # Tokenization, chat templates
├── llama.cpp/                 # llama.cpp submodule
│   ├── src/                   # Core implementation
│   ├── include/               # Headers
│   └── common/                # Common utilities
└── CMakeLists.txt             # Build configuration
```

---

## Core Components

### 1. Lifecycle Management (`jni_lifecycle.cpp`)

**Functions:**
- `loadModel(path)` - Loads GGUF model for text generation
- `loadReranker(path)` - Loads model for document reranking
- `freeModel(ptr)` - Releases native memory

**Key Implementation Details:**
- Context size: 4096 tokens (default)
- Batch size: 2048 tokens
- Reranker uses `LLAMA_POOLING_TYPE_RANK` for classification
- Embeddings enabled for reranker models

**Threading:**
- Uses pthread with 32MB stack to prevent stack overflow
- Critical for grammar parsing (needs large stack)

### 2. Text Generation (`jni_completion.cpp`)

**Functions:**
- `generateCompletion(ptr, prompt, grammar, callback)` - Main generation loop
- `infill(ptr, prefix, suffix, callback)` - FIM (Fill-In-Middle) completion

**Architecture:**
```cpp
// Sampler Chain Order (important!)
1. Grammar Sampler (if provided)    // Filters logits first
2. Top-K (k=40)                     // Keep top 40 tokens
3. Top-P (p=0.95)                   // Nucleus sampling
4. Temperature (t=0.8)              // Randomness
5. Distribution (random sampling)   // Final selection
```

**CRITICAL FIX - Grammar Handling:**
```cpp
// ❌ OLD (CRASHED): Manual validation after sampling
llama_token token = llama_sampler_sample(smpl, ctx, -1);
llama_sampler_accept(smpl, token);  // Threw exceptions!

// ✅ NEW (WORKS): Grammar sampler in chain handles everything
llama_token token = llama_sampler_sample(smpl, ctx, -1);
// No manual accept needed - grammar already enforced!
```

**Why this works:** The grammar sampler filters the probability distribution BEFORE sampling, ensuring only valid tokens can be selected. This matches llama-server behavior exactly.

### 3. Grammar System (`jni_grammar.cpp`)

**Functions:**
- `jsonSchemaToGrammar(schema)` - Converts JSON Schema to GBNF
- `loadGrammarFromFile(path)` - Loads GBNF from file
- `getGrammarInfo(grammar)` - Returns grammar statistics

**GBNF Grammar Format:**
```gbnf
root ::= "{" content "}"
content ::= [^}] content |  // Recursive - keeps expecting }
```

**PCRE2 Integration:**
- Replaced `std::regex` with PCRE2 to prevent Android stack overflow
- PCRE2 uses non-recursive NFA matching
- Safe on JNI threads with limited stack (~1MB)

**Grammar Validation:**
- Simple grammars: `root ::= "{" [^}]* "}"`
- Complex auto-generated: 226+ chars with nested rules
- Nightmare test: 1171 chars with arrays, nested objects

### 4. Embeddings (`jni_embedding.cpp`)

**Functions:**
- `getEmbedding(ptr, text)` - Returns float[] embedding vector

**Pooling Types:**
- `LLAMA_POOLING_TYPE_NONE` - Last token (generation)
- `LLAMA_POOLING_TYPE_MEAN` - Average (embeddings)
- `LLAMA_POOLING_TYPE_RANK` - Classification (reranking)

### 5. Reranking (`jni_rerank.cpp`)

**Functions:**
- `rerank(ptr, query, documents[])` - Returns relevance scores

**Architecture:**
1. Detects FIM tokens from vocab or uses fallback template
2. Sequential processing (one doc at a time)
3. Encoder/Decoder auto-detection:
   - BERT models: Uses `llama_encode`
   - Decoder models: Uses `llama_decode`
4. Score extraction from embeddings

### 6. Utilities (`jni_utils.cpp`)

**Functions:**
- `tokenize(ptr, text)` - Text → token IDs
- `detokenize(ptr, tokens)` - Token IDs → text
- `applyChatTemplate(ptr, roles, contents, customTemplate)` - Format chat

**Chat Template Detection:**
- Uses `llama_model_chat_template` for automatic detection
- Supports custom templates via parameter

### 7. Session Management (`jni_session.cpp`)

**Functions:**
- `saveSession(ptr, path)` - Saves KV cache to disk
- `loadSession(ptr, path)` - Restores KV cache

**Use Case:** Resume long conversations without reprocessing prompt.

### 8. LoRA Support (`jni_lora.cpp`)

**Functions:**
- `loadLora(ptr, path, scale)` - Load LoRA adapter
- `clearLoras(ptr)` - Remove all adapters

**Scaling:**
- Scale factor (0.0 - 1.0+) controls adapter strength
- Multiple adapters can be stacked

---

## Thread Safety

### pthread with 32MB Stack

```cpp
pthread_attr_t attr;
pthread_attr_init(&attr);
pthread_attr_setstacksize(&attr, 32 * 1024 * 1024);  // 32MB
pthread_create(&thread, &attr, generate_worker, &params);
```

**Why 32MB?**
- Android JNI threads default to ~1MB
- Grammar parsing uses PCRE2 which needs stack
- std::regex caused stack overflow on complex patterns

### Exception Handling

All generation loops wrapped in try-catch:
```cpp
try {
    // Generation logic
} catch (const std::exception& e) {
    LOGE("Generation error: %s", e.what());
    // Graceful exit instead of crash
}
```

---

## Kotlin API (LlamaEngine.kt)

### Core Methods

```kotlin
// Model Management
fun load(modelPath: String)
fun loadRerankModel(modelPath: String)
fun close()

// Generation
fun generate(prompt: String, grammar: String? = null): Flow<String>
fun infill(prefix: String, suffix: String): Flow<String>

// Embeddings & Reranking
fun embedding(text: String): FloatArray
fun rerank(query: String, documents: List<String>): List<Float>

// Tokenization
fun tokenize(text: String): IntArray
fun detokenize(tokens: IntArray): String

// Chat Templates
fun formatChat(messages: List<Pair<String, String>>, template: String? = null): String

// Grammar
fun convertJsonSchema(schema: String): String?
fun loadGrammarFile(path: String): String?
fun getGrammarStats(grammar: String): String?

// Session
fun saveState(path: String): Boolean
fun loadState(path: String): Boolean

// LoRA
fun addLora(path: String, scale: Float = 1.0f): Boolean
fun resetLoras()

// Metrics
fun getBench(): String
```

### Usage Example

```kotlin
val engine = LlamaEngine()
engine.load("/data/local/tmp/model.gguf")

// Simple generation
engine.generate("What is AI?").collect { token ->
    print(token)
}

// With grammar constraint
val grammar = """root ::= "{" content "}" 
content ::= [^}] content |"""
engine.generate("Create JSON", grammar).collect { token ->
    print(token)
}

// Cleanup
engine.close()
```

---

## Build System

### CMakeLists.txt Structure

```cmake
# PCRE2 for regex (Android-safe)
find_package(PkgConfig)
pkg_check_modules(PCRE2 libpcre2-8)

# llama.cpp
add_subdirectory(llama.cpp)

# JNI Library
add_library(cortex-engine SHARED
    jni/jni_*.cpp
)

target_link_libraries(cortex-engine
    llama
    ggml
    ${PCRE2_LIBRARIES}
)
```

### Key Build Flags

- `GGML_VULKAN=OFF` - Disabled (shader issues)
- `GGML_OPENMP=OFF` - Disabled (Android threading issues)
- `-O3 -march=armv8.2-a+fp16+dotprod` - ARM64 optimizations

---

## Dependencies

### System Requirements
- Android API 26+ (Android 8.0)
- ARM64 device (arm64-v8a)
- 4GB+ RAM recommended

### Libraries
- libllama.so - Core inference
- libggml.so - Tensor operations  
- libpcre2-8.so - Regex for grammar (Android-safe)

### Model Requirements
- GGUF format (llama.cpp native)
- Quantized models (Q4_K_M, Q5_K_M recommended)
- Chat models for chat templates

---

## Performance Considerations

### Memory Usage
- Model size: ~4-8GB for 7B parameters (Q4)
- Context: ~500MB for 4K context
- Total: ~6-10GB RAM recommended

### Speed
- ~2-10 tokens/second on modern phones (Pixel 7+, S23+)
- GPU acceleration via Vulkan (disabled in current build)
- Multi-threading via GGML CPU backend

### Optimization Tips
1. Use quantized models (Q4_K_M is sweet spot)
2. Reduce context size if not needed
3. Batch size affects memory but not speed much
4. Grammar adds ~10-20% overhead

---

## Limitations & Known Issues

1. **Vulkan GPU acceleration** - Disabled due to shader compilation issues
2. **Grammar complexity** - Very complex schemas may need simplification
3. **Memory** - Large models (70B+) won't fit on mobile devices
4. **Threading** - Uses CPU only (GPU via Vulkan experimental)

---

## Future Enhancements

- [ ] Vulkan/OpenCL GPU support
- [ ] Multimodal (vision) support via libmtmd
- [ ] Speculative decoding
- [ ] Quantization-aware training (QAT)
- [ ] Dynamic batching

---

## References

- llama.cpp: https://github.com/ggerganov/llama.cpp
- GGML: https://github.com/ggerganov/ggml
- PCRE2: https://www.pcre.org/
- JNI Documentation: https://docs.oracle.com/javase/8/docs/technotes/guides/jni/

---

## License

Same as llama.cpp (MIT) - See LICENSE file in repository.
