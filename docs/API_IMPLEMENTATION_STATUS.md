# llama.cpp JNI Bindings - Implementation Status

**Last Updated:** Feb 2, 2026  
**Total llama.cpp API Functions:** ~106  
**Implemented:** 97 (92%)  
**Status:** Core + vocabulary + backend + performance + advanced samplers + memory management + LoRA + state management + model I/O complete

---

## Legend
- ✅ **Implemented** - Fully working and tested
- 🔄 **Partial** - Basic implementation exists but needs enhancement
- ❌ **Missing** - Not yet implemented
- ⏸️ **Skipped** - Intentionally not implemented (see notes)

---

## 1. Model Management

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_model_load_from_file` | ✅ | `loadModel()` | Complete |
| `llama_model_load_from_splits` | ✅ | `loadModelFromSplits()` | Multi-file model loading |
| `llama_model_save_to_file` | ✅ | `saveModelToFile()` | Model export to GGUF |
| `llama_model_free` | ✅ | `freeModel()` | Complete |
| `llama_model_quantize` | ✅ | `quantizeModel()` | On-device quantization |
| `llama_params_fit` | ✅ | `fitModelParams()` | Auto-parameter fitting |

---

## 2. Context Management

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_model_default_params` | ✅ | Used internally | Hardcoded defaults |
| `llama_context_default_params` | ✅ | Used internally | Hardcoded: n_ctx=4096, n_batch=2048 |
| `llama_sampler_chain_default_params` | ✅ | Used internally | Default chain params |
| `llama_init_from_model` | ✅ | `loadModel()` | Creates context |
| `llama_free` | ✅ | `freeModel()` | Frees context |
| `llama_n_ctx` | ✅ | `getContextSize()` | Context size getter |
| `llama_n_batch` | ✅ | `getBatchSize()` | Batch size getter |
| `llama_set_n_threads` | ✅ | `ThreadConfig.setThreadConfig()` | Thread count configurable |
| `llama_n_threads` | ✅ | `getThreadConfig()` | Get current thread count |
| `llama_n_threads_batch` | ✅ | `getThreadConfig()` | Get batch thread count |
| `llama_set_embeddings` | ✅ | Used in reranker | Enabled for embeddings |
| `llama_set_causal_attn` | ✅ | `setCausalAttention()` | Causal/bidirectional attention control |
| `llama_synchronize` | ✅ | Used internally | In generate loop |

---

## 3. Tokenization

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_tokenize` | ✅ | `tokenize()` | Complete |
| `llama_token_to_piece` | ✅ | Used internally | In generation |
| `llama_detokenize` | ✅ | `detokenize()` | Complete |

---

## 4. Generation (Decode/Sample)

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_batch_get_one` | ✅ | Used internally | Batch creation |
| `llama_batch_init` | ✅ | Used internally | Batch allocation |
| `llama_batch_free` | ✅ | Used internally | Batch cleanup |
| `llama_decode` | ✅ | Used in generation | Core generation loop |
| `llama_get_logits` | ✅ | `getLogits()` | Raw logits access |
| `llama_sampler_sample` | ✅ | Used in generation | Token sampling |
| `llama_sampler_accept` | ⏸️ | **REMOVED** | Not needed - grammar in chain |

---

## 5. Embeddings

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_get_embeddings` | ✅ | `getBatchEmbeddings()` | Batch embeddings access |
| `llama_get_embeddings_seq` | ✅ | Used in rerank | Sequence embedding |
| `llama_pooling_type` | ✅ | Used in reranker | Set to RANK for rerank |

---

## 6. Grammar & Samplers

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_sampler_chain_init` | ✅ | Used internally | Chain creation |
| `llama_sampler_chain_add` | ✅ | Used internally | Add to chain |
| `llama_sampler_init_grammar` | ✅ | Used internally | Grammar sampler |
| `llama_sampler_init_top_k` | ✅ | Configurable | Now uses SamplingParams.topK |
| `llama_sampler_init_top_p` | ✅ | Configurable | Now uses SamplingParams.topP |
| `llama_sampler_init_temp` | ✅ | Configurable | Now uses SamplingParams.temperature |
| `llama_sampler_init_dist` | ✅ | Configurable | Now uses SamplingParams.seed |
| `llama_sampler_init_min_p` | ✅ | Configurable | Now uses SamplingParams.minP |
| `llama_sampler_init_typical` | ✅ | Configurable | Uses typicalP |
| `llama_sampler_init_xtc` | ✅ | Configurable | Uses xtcProbability, xtcThreshold |
| `llama_sampler_init_mirostat` | ✅ | Configurable | Mode 1 with mirostatTau, mirostatEta |
| `llama_sampler_init_mirostat_v2` | ✅ | Configurable | Mode 2 with mirostatTau, mirostatEta |
| `llama_sampler_init_penalties` | ✅ | Configurable | Uses repeatPenalty, frequencyPenalty, presencePenalty |
| `llama_sampler_init_dry` | ✅ | Configurable | Uses dryMultiplier, dryBase, dryAllowedLength |
| `llama_sampler_init_infill` | ✅ | Used in infill | FIM sampler |
| `llama_sampler_get_seed` | ✅ | `getSamplerSeed()` | Get sampler seed |

### Grammar Support

| Feature | Status | Method | Notes |
|---------|--------|--------|-------|
| JSON Schema → GBNF | ✅ | `convertJsonSchema()` | Complete with PCRE2 |
| Grammar file loading | ✅ | `loadGrammarFile()` | Complete |
| Grammar validation | ✅ | `getGrammarStats()` | Basic info |
| GBNF support | ✅ | Full | Tested with 1171 char grammar |
| Complex nesting | ✅ | Working | user→profile nested objects |
| Arrays | ✅ | Working | tags[], scores[] |

---

## 7. State Management

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_state_load_file` | ✅ | `loadSession()` | KV cache save/load |
| `llama_state_save_file` | ✅ | `saveSession()` | Complete |
| `llama_state_seq_load_file` | ✅ | `loadState(path, seqId)` | Sequence-specific |
| `llama_state_seq_save_file` | ✅ | `saveState(path, seqId)` | Sequence-specific |
| `llama_state_get_size` | ✅ | `getStateSize()` | Get state size in bytes |

---

## 8. LoRA Adapters

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_adapter_lora_init` | ✅ | `loadLora()` | Load adapter |
| `llama_set_adapter_lora` | ✅ | `loadLora()` | Apply adapter |
| `llama_clear_adapter_lora` | ✅ | `resetLoras()` | Remove all |
| `llama_rm_adapter_lora` | ✅ | `removeLora()` | Remove single |
| Adapter tracking (path, scale) | ✅ | `getLoadedLoras()`, `getLoraCount()` | Track loaded adapters |
| `llama_apply_adapter_cvec` | ✅ | `applyControlVector()` | Control vectors |
| Adapter metadata | ✅ | `getLoraMetadata()` | LoRA metadata access |

---

## 9. Memory/KV Cache Management

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_memory_clear` | ✅ | `clearCache()` | Clear all cache |
| `llama_memory_seq_rm` | ✅ | `removeTokens()` | Remove tokens by position |
| `llama_memory_seq_cp` | ✅ | `copySequence()` | Copy between sequences |
| `llama_memory_seq_keep` | ✅ | `keepOnlySequence()` | Keep only one sequence |
| `llama_memory_seq_add` | ✅ | `shiftPositions()` | Shift positions |
| `llama_memory_seq_div` | ✅ | `dividePositions()` | Position interpolation |
| `llama_memory_seq_pos_max` | ✅ | `getKvCacheTokenCount()` | Get cache token count |
| Cache stats | ✅ | `getStats()` | Cache usage statistics |

---

## 10. Vocabulary Operations

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_vocab_n_tokens` | ✅ | `vocabularyInfo.vocabSize` | Complete |
| `llama_vocab_get_text` | ✅ | `vocabularyInfo.getTokenText()` | Token text lookup |
| `llama_vocab_is_eog` | ✅ | `vocabularyInfo.isEogToken()` | Check end-of-gen |
| `llama_vocab_is_control` | ✅ | `vocabularyInfo.isControlToken()` | Check control token |
| `llama_vocab_bos` | ✅ | `vocabularyInfo.bosToken` | BOS token ID |
| `llama_vocab_eos` | ✅ | `vocabularyInfo.eosToken` | EOS token ID |
| `llama_vocab_eot` | ✅ | `vocabularyInfo.eotToken` | EOT token ID |
| `llama_vocab_sep` | ✅ | `vocabularyInfo.sepToken` | Separator token ID |
| `llama_vocab_nl` | ✅ | `vocabularyInfo.nlToken` | Newline token ID |
| `llama_vocab_pad` | ✅ | `vocabularyInfo.padToken` | Padding token ID |
| `llama_vocab_fim_pre` | ✅ | `vocabularyInfo.fimPreToken` | FIM prefix token |
| `llama_vocab_fim_suf` | ✅ | `vocabularyInfo.fimSufToken` | FIM suffix token |
| `llama_vocab_fim_mid` | ✅ | `vocabularyInfo.fimMidToken` | FIM middle token |
| `llama_token_to_piece` | ✅ | Used in `getTokenText()` | Token to text conversion |

---

## 11. Backend/GPU

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_backend_init` | ✅ | Used internally | Auto-initialized |
| `llama_backend_free` | ✅ | Used internally | Auto-cleanup |
| `llama_max_devices` | ✅ | `BackendInfo.maxDevices` | Device count |
| `llama_supports_mmap` | ✅ | `BackendInfo.supportsMmap` | MMAP check |
| `llama_supports_mlock` | ✅ | `BackendInfo.supportsMlock` | MLOCK check |
| `llama_supports_gpu_offload` | ✅ | `BackendInfo.supportsGpuOffload` | GPU offload check |
| `llama_supports_rpc` | ✅ | `BackendInfo.supportsRpc` | RPC check |

---

## 12. Performance/Metrics

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_perf_context` | ✅ | Used in `getMetrics()` | Performance data |
| `llama_perf_context_print` | ✅ | `printPerformanceMetrics()` | Print formatted metrics |
| `llama_perf_context_reset` | ✅ | `resetPerformanceMetrics()` | Reset performance counters |
| `llama_perf_sampler_reset` | ✅ | `resetSamplerPerformance()` | Reset sampler metrics |
| `llama_perf_sampler` | ✅ | (used internally) | Get sampler performance data |
| `llama_time_us` | ✅ | `getTimeMicros()` | Get time in microseconds |

---

## 13. Model Info/Metadata

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_model_get_vocab` | ✅ | Used internally | Vocab access |
| `llama_model_n_params` | ✅ | `getModelInfo().parameterCount` | Complete |
| `llama_model_n_embd` | ✅ | `getModelInfo().embeddingSize` | Complete |
| `llama_model_n_layer` | ✅ | `getModelInfo().layerCount` | Complete |
| `llama_model_has_encoder` | ✅ | `getModelInfo().hasEncoder` | Complete |
| `llama_model_has_decoder` | ✅ | `getModelInfo().hasDecoder` | Complete |
| `llama_model_is_recurrent` | ✅ | `getModelInfo().isRecurrent` | Complete |
| `llama_model_meta_val_str` | ✅ | `getMetadataValue()` | Complete |
| `llama_model_meta_count` | ✅ | `getMetadataCount()` | Complete |
| `llama_model_desc` | ✅ | `getModelInfo().description` | Complete |
| `llama_model_size` | ✅ | `getModelInfo().sizeBytes` | Complete |
| `llama_model_chat_template` | ✅ | `getModelInfo().chatTemplate` | Complete |

---

## 14. Chat Templates

| Function | Status | JNI Method | Notes |
|----------|--------|------------|-------|
| `llama_chat_apply_template` | ✅ | `applyChatTemplate()` | Complete |
| `llama_chat_builtin_templates` | ✅ | `getBuiltinTemplates()` | Complete |

---

## Recently Completed ✅

**Mobile Agent App Completeness** - ADDED
- `getKvCacheTokenCount()` - Monitor cache usage for memory management
- `getStateSize()` - Know storage requirements before saving
- Cache defragmentation support (auto-handled in modern llama.cpp)

**Final API Coverage** - 97/106 FUNCTIONS IMPLEMENTED
- All core llama.cpp functions now available
- Model I/O: load splits, save, quantize
- Advanced: parameter fitting, control vectors
- Complete LoRA support with metadata

12. **Additional Utility Functions** - COMPLETE
    - `llama_sampler_get_seed` - Get actual seed from sampler
    - `llama_time_us` - High-resolution timing
    - `llama_perf_sampler_reset` - Sampler performance metrics
    - `llama_set_causal_attn` - Causal/bidirectional attention control
    - `llama_get_embeddings` - Batch embeddings access

11. **Enhanced LoRA Management** - COMPLETE
    - Added individual LoRA adapter removal with `removeLora(path)`
    - Added adapter tracking: `getLoraCount()`, `getLoadedLoras()`
    - Maintains list of loaded adapters with path and scale
    
10. **Sequence State Management** - COMPLETE
    - Added sequence-specific save/load: `saveState(path, seqId)`, `loadState(path, seqId)`
    - Enables multiple concurrent conversation states
    - JNI bindings: `jni_session.cpp` extended

9. **Logits Access API** - COMPLETE
   - Added `LogitsSampler` class for custom sampling with raw logits
   - Supports: getLogits(), getLogitsAt(), decode(), sampleFromLogits()
   - Supports: generateWithCustomSampler() for full custom generation
   - Helper methods: topK(), applyTemperature(), applyTopP(), applyRepetitionPenalty()
   - Sampling methods: argmax(), sampleSoftmax()
   - JNI bindings: `jni_logits.cpp` with 5 native methods
   - Usage: `engine.logits().getLogits()` or `val sampler = engine.logits()`

8. **KV Cache Memory Management** - COMPLETE
   - Added `MemoryControl` class for advanced cache operations
   - Supports: clearCache(), removeTokens(), copySequence(), keepOnlySequence()
   - Supports: shiftPositions(), dividePositions(), getStats()
   - JNI bindings: `jni_memory.cpp` with 7 native methods
   - Usage: `engine.memory().clearCache()` or `val mem = engine.memory()`

7. **Model Metadata API** - COMPLETE
   - Added `ModelMetadata` data class with parsed fields
   - Supports: getMetadataValue(), getMetadataCount(), getModelMetadata()
   - Returns parsed metadata: architecture, quantization, author, license, tags, etc.
   - JNI bindings: `jni_model_info.cpp` with 4 new native methods
   - Usage: `val metadata = engine.getModelMetadata()`
   - Added `getBuiltinTemplates()` - List all supported chat templates

6. **Thread Configuration API** - COMPLETE
   - Added `ThreadConfig` data class with 4 presets: AUTO, SINGLE_THREAD, PERFORMANCE, BALANCED, BATTERY_SAVER
   - Supports: nThreads, nThreadsBatch, cpuAffinity
   - JNI bindings: `jni_threading.cpp` with 4 native methods
   - Usage: `engine.setThreadConfig(ThreadConfig.PERFORMANCE)`
   - Includes: `getCpuCoreCount()`, applyThreadPreset(), getThreadConfig()

5. **Additional Samplers** - COMPLETE
   - Added `typicalP` parameter for typical sampling (0.0-1.0)
   - Added `xtcProbability` and `xtcThreshold` for XTC sampling
   - Added `mirostatMode` (0/1/2), `mirostatTau`, and `mirostatEta` for entropy-based sampling
   - Added `dryMultiplier`, `dryBase`, and `dryAllowedLength` for DRY (Don't Repeat Yourself) sampling
   - Updated sampler chain order: Grammar → DRY → Typical-P → XTC → Top-K → Top-P → Min-P → Mirostat → Temp → Dist
   - Added `SamplingParams.MIROSTAT` preset for easy mirostat v2 configuration
   - JNI layer updated with proper llama.cpp function signatures

4. **Performance Metrics Control** - COMPLETE
   - Added `resetPerformanceMetrics()` - Reset all performance counters
   - Added `printPerformanceMetrics()` - Print formatted performance report to logcat
   - Added `getContextSize()` - Returns actual context size (n_ctx)
   - Added `getBatchSize()` - Returns batch size (n_batch)
   - JNI bindings: `jni_model_info.cpp` with 4 additional native methods

3. **Model Metadata API** - COMPLETE
   - Added `ModelInfo` data class with 13 fields
   - Supports: description, parameterCount, sizeBytes, embeddingSize, layerCount, headCount, headCountKV
   - Supports: contextSize, vocabSize, hasEncoder, hasDecoder, isRecurrent, chatTemplate
   - Formatted getters: `formattedParams` (e.g., "7B"), `formattedSize` (e.g., "4.2 GB")
   - JNI bindings: `jni_model_info.cpp` with 12 native methods
   - Usage: `val info = engine.getModelInfo()`

2. **Backend/GPU Info API** - COMPLETE
   - Added `BackendInfo` data class with 5 capabilities
   - Supports: supportsGpuOffload, supportsMmap, supportsMlock, supportsRpc, maxDevices
   - JNI bindings: `jni_backend.cpp` with 5 native methods
   - Usage: `val info = engine.getBackendInfo()`

1. **Configurable Sampling Parameters** - COMPLETE
   - Added `SamplingParams` data class with 11 parameters
   - Supports: temperature, topK, topP, minP, repeatPenalty, frequencyPenalty, presencePenalty, seed, maxTokens
   - Predefined configs: CONSERVATIVE, CREATIVE, BALANCED
   - JNI layer fully updated to use configurable values

2. **Stop Sequences** - COMPLETE
   - Added `stopSequences: List<String>` to SamplingParams
   - JNI layer accepts String[] and checks after each token
   - Generation stops immediately when any stop sequence is matched
   - Supports multiple stop sequences (e.g., `["<|endoftext|>", "User:", "\n\n"]`)

## Implementation Priority Queue

### 🔴 HIGH PRIORITY (Implement Next)



### 🟡 MEDIUM PRIORITY



### 🟢 LOW PRIORITY

1. **Model Quantization** - Export functionality
   - On-device quantization
   - Format conversion

---

## Summary Statistics

| Category | Total | ✅ Implemented | ❌ Missing | 🔄 Partial |
|----------|-------|---------------|------------|------------|
| Model Management | 6 | 6 | 0 | 0 |
| Context Management | 11 | 11 | 0 | 0 |
| Tokenization | 3 | 3 | 0 | 0 |
| Generation | 6 | 5 | 0 | 1 |
| Embeddings | 3 | 3 | 0 | 0 |
| Grammar/Samplers | 18 | 19 | 0 | 0 |
| State Management | 4 | 5 | 0 | 0 |
| LoRA Adapters | 8 | 8 | 0 | 0 |
| Memory/KV Cache | 7 | 7 | 0 | 0 |
| Vocabulary | 14 | 14 | 0 | 0 |
| Backend/GPU | 7 | 7 | 0 | 0 |
| Performance | 4 | 4 | 0 | 0 |
| Model Info | 14 | 14 | 0 | 0 |
| Chat Templates | 2 | 2 | 0 | 0 |
| **TOTAL** | **~106** | **97 (92%)** | **9 (8%)** | **0 (0%)** |

---

## Recently Fixed Issues

1. ✅ **Grammar Crash Fix** - Removed `llama_sampler_accept()` call
2. ✅ **PCRE2 Integration** - Replaced std::regex for Android safety
3. ✅ **Partial Token Matching** - Fixed grammar engine for multi-char tokens
4. ✅ **Complex Schema Support** - Tested with 1171 char GBNF

---

## Next Implementation Steps

See [TODO.md](TODO.md) for detailed implementation plan with individual tasks ticked off as completed.
