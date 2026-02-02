package com.cortex.app

import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import kotlinx.coroutines.Dispatchers

class LlamaEngine {

    interface CompletionCallback {
        fun onToken(tokenBytes: ByteArray)
    }

    private var contextPointer: Long = 0

    companion object {
        init {
            System.loadLibrary("cortex-engine")
        }
    }

    private external fun loadModel(modelPath: String): Long
    private external fun loadReranker(modelPath: String): Long
    private external fun freeModel(contextPtr: Long)
    private external fun generateCompletion(
        contextPtr: Long, 
        prompt: String, 
        grammarJson: String?,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repeatPenalty: Float,
        repeatLastN: Int,
        frequencyPenalty: Float,
        presencePenalty: Float,
        seed: Int,
        maxTokens: Int,
        stopSequences: Array<String>,
        typicalP: Float,
        xtcProbability: Float,
        xtcThreshold: Float,
        mirostatMode: Int,
        mirostatTau: Float,
        mirostatEta: Float,
        dryMultiplier: Float,
        dryBase: Float,
        dryAllowedLength: Int,
        callback: CompletionCallback
    )
    private external fun infill(
        contextPtr: Long,
        prefix: String,
        suffix: String,
        callback: CompletionCallback
    )
    private external fun getEmbedding(contextPtr: Long, text: String): FloatArray?
    private external fun rerank(contextPtr: Long, query: String, documents: Array<String>): FloatArray?
    private external fun tokenize(contextPtr: Long, text: String): IntArray?
    private external fun detokenize(contextPtr: Long, tokens: IntArray): String?
    private external fun applyChatTemplate(contextPtr: Long, roles: Array<String>, contents: Array<String>, customTemplate: String?): String?
    private external fun jsonSchemaToGrammar(jsonSchema: String): String?
    private external fun loadGrammarFromFile(filePath: String): String?
    private external fun getGrammarInfo(grammarStr: String): String?
    
    // New Bindings
    private external fun saveSession(contextPtr: Long, path: String): Boolean
    private external fun loadSession(contextPtr: Long, path: String): Boolean
    private external fun loadLora(contextPtr: Long, path: String, scale: Float): Boolean
    private external fun clearLoras(contextPtr: Long)
    private external fun getMetrics(contextPtr: Long): String?
    
    // Enhanced LoRA Management
    internal external fun removeLora(contextPtr: Long, path: String): Boolean
    internal external fun getLoraCount(contextPtr: Long): Int
    internal external fun getLoadedLoras(contextPtr: Long): Array<String>?

    // Sequence-specific State Management
    internal external fun saveSessionSequence(contextPtr: Long, path: String, seqId: Int): Boolean
    internal external fun loadSessionSequence(contextPtr: Long, path: String, seqId: Int): Boolean
    
    // Model Info Bindings
    private external fun getModelDescription(contextPtr: Long): String?
    private external fun getModelParameterCount(contextPtr: Long): Long
    private external fun getModelSize(contextPtr: Long): Long
    private external fun getModelEmbeddingSize(contextPtr: Long): Int
    private external fun getModelLayerCount(contextPtr: Long): Int
    private external fun getModelHeadCount(contextPtr: Long): Int
    private external fun getModelHeadCountKV(contextPtr: Long): Int
    private external fun getModelContextSize(contextPtr: Long): Int
    private external fun getModelVocabSize(contextPtr: Long): Int
    private external fun modelHasEncoder(contextPtr: Long): Boolean
    private external fun modelHasDecoder(contextPtr: Long): Boolean
    private external fun modelIsRecurrent(contextPtr: Long): Boolean
    private external fun getModelChatTemplate(contextPtr: Long): String?

    // Model Metadata
    internal external fun getModelMetadataCount(contextPtr: Long): Int
    internal external fun getModelMetadataValue(contextPtr: Long, key: String): String?
    internal external fun getAllModelMetadata(contextPtr: Long): String?
    internal external fun getBuiltinChatTemplates(): String?

    // Vocabulary Bindings
    internal external fun getVocabSize(contextPtr: Long): Int
    internal external fun getTokenText(contextPtr: Long, token: Int): String?
    internal external fun isEogToken(contextPtr: Long, token: Int): Boolean
    internal external fun isControlToken(contextPtr: Long, token: Int): Boolean
    internal external fun getBosToken(contextPtr: Long): Int
    internal external fun getEosToken(contextPtr: Long): Int
    internal external fun getEotToken(contextPtr: Long): Int
    internal external fun getSepToken(contextPtr: Long): Int
    internal external fun getNlToken(contextPtr: Long): Int
    internal external fun getPadToken(contextPtr: Long): Int
    internal external fun getFimPreToken(contextPtr: Long): Int
    internal external fun getFimSufToken(contextPtr: Long): Int
    internal external fun getFimMidToken(contextPtr: Long): Int
    
    // Context Getters
    internal external fun getContextSize(contextPtr: Long): Int
    internal external fun getBatchSize(contextPtr: Long): Int
    
    // KV Cache Memory Management
    internal external fun nativeClearKvCache(contextPtr: Long, clearData: Boolean)
    internal external fun nativeRemoveKvCacheTokens(contextPtr: Long, seqId: Int, pos0: Int, pos1: Int): Boolean
    internal external fun nativeCopyKvCacheSequence(contextPtr: Long, srcSeqId: Int, dstSeqId: Int, pos0: Int, pos1: Int)
    internal external fun nativeKeepKvCacheSequence(contextPtr: Long, seqId: Int)
    internal external fun nativeShiftKvCachePositions(contextPtr: Long, seqId: Int, pos0: Int, pos1: Int, delta: Int)
    internal external fun nativeDivideKvCachePositions(contextPtr: Long, seqId: Int, pos0: Int, pos1: Int, divisor: Int)
    internal external fun nativeGetKvCacheStats(contextPtr: Long): IntArray
    
    // Logits Access
    internal external fun nativeGetLogits(contextPtr: Long): FloatArray?
    internal external fun nativeGetLogitsIth(contextPtr: Long, index: Int): FloatArray?
    internal external fun nativeGetVocabSizeFromContext(contextPtr: Long): Int
    internal external fun nativeDecodeTokens(contextPtr: Long, tokens: IntArray): Boolean
    internal external fun nativeSampleTokenFromLogits(contextPtr: Long, logits: FloatArray, temperature: Float): Int
    
    // Performance Control
    internal external fun resetPerformanceMetrics(contextPtr: Long)
    internal external fun printPerformanceMetrics(contextPtr: Long): String?
    
    // Thread Configuration
    internal external fun nativeSetThreadConfig(contextPtr: Long, nThreads: Int, nThreadsBatch: Int, cpuAffinity: Boolean)
    internal external fun nativeGetThreadConfig(contextPtr: Long): IntArray
    internal external fun nativeGetCpuCoreCount(): Int
    internal external fun nativeApplyThreadPreset(contextPtr: Long, preset: Int)
    
    internal fun getContextPointer(): Long = contextPointer

    fun load(modelPath: String) {
        if (contextPointer != 0L) {
            freeModel(contextPointer)
        }
        contextPointer = loadModel(modelPath)
        if (contextPointer == 0L) {
            throw RuntimeException("Failed to load model at $modelPath")
        }
    }

    fun loadRerankModel(modelPath: String) {
        if (contextPointer != 0L) {
            freeModel(contextPointer)
        }
        contextPointer = loadReranker(modelPath)
        if (contextPointer == 0L) {
            throw RuntimeException("Failed to load reranker at $modelPath")
        }
    }

    fun getBench(): String {
        if (contextPointer == 0L) return "{}"
        return getMetrics(contextPointer) ?: "{}"
    }
    
    /**
     * Get the actual context size (number of tokens the context can hold)
     */
    fun getContextSize(): Int {
        if (contextPointer == 0L) return 0
        return getContextSize(contextPointer)
    }
    
    /**
     * Get the batch size (number of tokens processed in parallel)
     */
    fun getBatchSize(): Int {
        if (contextPointer == 0L) return 0
        return getBatchSize(contextPointer)
    }
    
    /**
     * Reset performance metrics counters
     */
    fun resetPerformanceMetrics() {
        if (contextPointer != 0L) {
            resetPerformanceMetrics(contextPointer)
        }
    }
    
    /**
     * Get formatted performance metrics as a string
     */
    fun printPerformanceMetrics(): String {
        if (contextPointer == 0L) return "No model loaded"
        return printPerformanceMetrics(contextPointer) ?: "No metrics available"
    }
    
    fun getModelInfo(): ModelInfo {
        if (contextPointer == 0L) return ModelInfo.EMPTY
        
        return ModelInfo(
            description = getModelDescription(contextPointer) ?: "Unknown",
            parameterCount = getModelParameterCount(contextPointer),
            sizeBytes = getModelSize(contextPointer),
            embeddingSize = getModelEmbeddingSize(contextPointer),
            layerCount = getModelLayerCount(contextPointer),
            headCount = getModelHeadCount(contextPointer),
            headCountKV = getModelHeadCountKV(contextPointer),
            contextSize = getModelContextSize(contextPointer),
            vocabSize = getModelVocabSize(contextPointer),
            hasEncoder = modelHasEncoder(contextPointer),
            hasDecoder = modelHasDecoder(contextPointer),
            isRecurrent = modelIsRecurrent(contextPointer),
            chatTemplate = getModelChatTemplate(contextPointer)
        )
    }

    fun getVocabulary(): VocabularyInfo {
        return VocabularyInfo.from(this)
    }

    fun rerank(query: String, documents: List<String>): List<Float> {
        if (contextPointer == 0L) throw IllegalStateException("Model not loaded")
        val scores = rerank(contextPointer, query, documents.toTypedArray())
        return scores?.toList() ?: emptyList()
    }

    fun generate(
        prompt: String, 
        grammar: String? = null,
        params: SamplingParams = SamplingParams()
    ): Flow<String> = callbackFlow {
        if (contextPointer == 0L) {
            close(IllegalStateException("Model not loaded"))
            return@callbackFlow
        }

        val callback = object : CompletionCallback {
            override fun onToken(tokenBytes: ByteArray) {
                // Convert bytes to string safely (replacing invalid chars)
                val token = String(tokenBytes, Charsets.UTF_8)
                trySend(token)
            }
        }

        withContext(Dispatchers.IO) {
            generateCompletion(
                contextPointer, 
                prompt, 
                grammar,
                params.temperature,
                params.topK,
                params.topP,
                params.minP,
                params.repeatPenalty,
                params.repeatLastN,
                params.frequencyPenalty,
                params.presencePenalty,
                params.seed,
                params.maxTokens,
                params.stopSequences.toTypedArray(),
                params.typicalP,
                params.xtcProbability,
                params.xtcThreshold,
                params.mirostatMode,
                params.mirostatTau,
                params.mirostatEta,
                params.dryMultiplier,
                params.dryBase,
                params.dryAllowedLength,
                callback
            )
        }
        
        close()
        awaitClose { }
    }

    fun infill(prefix: String, suffix: String): Flow<String> = callbackFlow {
        if (contextPointer == 0L) {
            close(IllegalStateException("Model not loaded"))
            return@callbackFlow
        }

        val callback = object : CompletionCallback {
            override fun onToken(tokenBytes: ByteArray) {
                val token = String(tokenBytes, Charsets.UTF_8)
                trySend(token)
            }
        }

        withContext(Dispatchers.IO) {
            infill(contextPointer, prefix, suffix, callback)
        }
        
        close()
        awaitClose { }
    }

    fun embedding(text: String): FloatArray {
        if (contextPointer == 0L) throw IllegalStateException("Model not loaded")
        return getEmbedding(contextPointer, text) ?: floatArrayOf()
    }

    fun tokenize(text: String): IntArray {
        if (contextPointer == 0L) throw IllegalStateException("Model not loaded")
        return tokenize(contextPointer, text) ?: intArrayOf()
    }

    fun detokenize(tokens: IntArray): String {
        if (contextPointer == 0L) throw IllegalStateException("Model not loaded")
        return detokenize(contextPointer, tokens) ?: ""
    }

    fun formatChat(messages: List<Pair<String, String>>, customTemplate: String? = null): String {
        if (contextPointer == 0L) throw IllegalStateException("Model not loaded")
        val roles = messages.map { it.first }.toTypedArray()
        val contents = messages.map { it.second }.toTypedArray()
        return applyChatTemplate(contextPointer, roles, contents, customTemplate) ?: ""
    }

    fun convertJsonSchema(jsonSchema: String): String? {
        return jsonSchemaToGrammar(jsonSchema)
    }

    fun loadGrammarFile(filePath: String): String? {
        return loadGrammarFromFile(filePath)
    }

    fun getGrammarStats(grammar: String): String? {
        return getGrammarInfo(grammar)
    }

    fun saveState(path: String): Boolean {
        if (contextPointer == 0L) return false
        return saveSession(contextPointer, path)
    }

    fun loadState(path: String): Boolean {
        if (contextPointer == 0L) return false
        return loadSession(contextPointer, path)
    }

    fun addLora(path: String, scale: Float): Boolean {
        if (contextPointer == 0L) return false
        return loadLora(contextPointer, path, scale)
    }

    fun resetLoras() {
        if (contextPointer != 0L) clearLoras(contextPointer)
    }

    /**
     * Remove a specific LoRA adapter by path.
     * 
     * @param path Path to the LoRA adapter file
     * @return true if successfully removed
     */
    fun removeLora(path: String): Boolean {
        if (contextPointer == 0L) return false
        return removeLora(contextPointer, path)
    }

    /**
     * Get the number of currently loaded LoRA adapters.
     */
    fun getLoraCount(): Int {
        if (contextPointer == 0L) return 0
        return getLoraCount(contextPointer)
    }

    /**
     * Get list of paths of all loaded LoRA adapters.
     */
    fun getLoadedLoras(): List<String> {
        if (contextPointer == 0L) return emptyList()
        val array = getLoadedLoras(contextPointer)
        return array?.toList() ?: emptyList()
    }

    /**
     * Save state for a specific sequence.
     * 
     * @param path File path to save to
     * @param seqId Sequence ID to save
     * @return true if successful
     */
    fun saveState(path: String, seqId: Int): Boolean {
        if (contextPointer == 0L) return false
        return saveSessionSequence(contextPointer, path, seqId)
    }

    /**
     * Load state for a specific sequence.
     * 
     * @param path File path to load from
     * @param seqId Sequence ID to load into
     * @return true if successful
     */
    fun loadState(path: String, seqId: Int): Boolean {
        if (contextPointer == 0L) return false
        return loadSessionSequence(contextPointer, path, seqId)
    }

    // Thread Configuration APIs
    
    /**
     * Apply thread configuration to control CPU usage.
     * 
     * @param config Thread configuration (use ThreadConfig.AUTO, PERFORMANCE, etc.)
     */
    fun setThreadConfig(config: ThreadConfig) {
        if (contextPointer == 0L) throw IllegalStateException("Model not loaded")
        nativeSetThreadConfig(contextPointer, config.nThreads, config.nThreadsBatch, config.cpuAffinity)
    }
    
    /**
     * Get current thread configuration.
     * 
     * @return Current thread configuration
     */
    fun getThreadConfig(): ThreadConfig {
        if (contextPointer == 0L) return ThreadConfig.AUTO
        val config = nativeGetThreadConfig(contextPointer)
        return ThreadConfig(
            nThreads = config[0],
            nThreadsBatch = config[1],
            cpuAffinity = config[2] == 1
        )
    }
    
    /**
     * Get the number of CPU cores available on the device.
     * Useful for determining optimal thread counts.
     */
    fun getCpuCoreCount(): Int {
        return nativeGetCpuCoreCount()
    }
    
    /**
     * Apply a thread configuration preset.
     * 
     * @param preset One of ThreadConfig.Preset values
     */
    fun applyThreadPreset(preset: Int) {
        if (contextPointer == 0L) throw IllegalStateException("Model not loaded")
        nativeApplyThreadPreset(contextPointer, preset)
    }
    
    /**
     * Thread configuration presets for convenience.
     */
    object ThreadPreset {
        const val AUTO = 0
        const val SINGLE = 1
        const val PERFORMANCE = 2
        const val BALANCED = 3
        const val BATTERY = 4
    }

    /**
     * Get model metadata (GGUF key-value pairs).
     * 
     * @return ModelMetadata object containing all metadata
     */
    fun getModelMetadata(): ModelMetadata {
        if (contextPointer == 0L) return ModelMetadata.EMPTY
        
        // Get all metadata as JSON and parse
        val jsonStr = getAllModelMetadata(contextPointer) ?: "{}"
        
        // Parse JSON to map (simple parsing)
        val metadata = parseMetadataJson(jsonStr)
        return ModelMetadata.parse(metadata)
    }

    /**
     * Get a specific metadata value by key.
     * 
     * @param key The metadata key (e.g., "general.architecture")
     * @return The metadata value, or null if not found
     */
    fun getMetadataValue(key: String): String? {
        if (contextPointer == 0L) return null
        return getModelMetadataValue(contextPtr = contextPointer, key)
    }

    /**
     * Get the number of metadata entries.
     */
    fun getMetadataCount(): Int {
        if (contextPointer == 0L) return 0
        return getModelMetadataCount(contextPointer)
    }

    /**
     * Get list of built-in chat templates.
     * 
     * @return List of template names supported by llama.cpp
     */
    fun getBuiltinTemplates(): List<String> {
        val json = getBuiltinChatTemplates() ?: "[]"
        // Parse JSON array
        return json.removeSurrounding("[", "]")
            .split(",")
            .map { it.trim().removeSurrounding("\"") }
            .filter { it.isNotEmpty() }
    }

    /**
     * Get a MemoryControl instance for managing the KV cache.
     *
     * @return MemoryControl instance
     */
    fun memory(): MemoryControl {
        if (contextPointer == 0L) throw IllegalStateException("Model not loaded")
        return MemoryControl(this)
    }

    /**
     * Get a LogitsSampler instance for custom sampling with raw logits access.
     * 
     * @return LogitsSampler instance
     */
    fun logits(): LogitsSampler {
        if (contextPointer == 0L) throw IllegalStateException("Model not loaded")
        return LogitsSampler(this)
    }

    /**
     * Simple JSON parser for metadata (since we don't have a JSON library dependency).
     */
    private fun parseMetadataJson(json: String): Map<String, String> {
        val result = mutableMapOf<String, String>()
        
        // Remove outer braces
        val content = json.removeSurrounding("{", "}").trim()
        if (content.isEmpty()) return result
        
        // Simple state machine to parse key:value pairs
        var i = 0
        while (i < content.length) {
            // Skip whitespace and commas
            while (i < content.length && (content[i].isWhitespace() || content[i] == ',')) {
                i++
            }
            if (i >= content.length) break
            
            // Parse key (must be in quotes)
            if (content[i] != '"') {
                i++
                continue
            }
            val keyStart = ++i
            while (i < content.length && content[i] != '"') {
                if (content[i] == '\\' && i + 1 < content.length) {
                    i += 2 // Skip escaped character
                } else {
                    i++
                }
            }
            val key = content.substring(keyStart, i)
            i++ // Skip closing quote
            
            // Skip to colon
            while (i < content.length && content[i] != ':') i++
            if (i >= content.length) break
            i++ // Skip colon
            
            // Skip whitespace
            while (i < content.length && content[i].isWhitespace()) i++
            
            // Parse value (must be in quotes)
            if (i >= content.length || content[i] != '"') {
                continue
            }
            val valStart = ++i
            val valueBuilder = StringBuilder()
            while (i < content.length && content[i] != '"') {
                if (content[i] == '\\' && i + 1 < content.length) {
                    // Handle escape sequences
                    when (content[i + 1]) {
                        'n' -> valueBuilder.append('\n')
                        'r' -> valueBuilder.append('\r')
                        't' -> valueBuilder.append('\t')
                        '\\' -> valueBuilder.append('\\')
                        '"' -> valueBuilder.append('"')
                        else -> valueBuilder.append(content[i + 1])
                    }
                    i += 2
                } else {
                    valueBuilder.append(content[i])
                    i++
                }
            }
            val value = valueBuilder.toString()
            i++ // Skip closing quote
            
            result[key] = value
        }
        
        return result
    }

    fun close() {
        if (contextPointer != 0L) {
            freeModel(contextPointer)
            contextPointer = 0
        }
    }
}