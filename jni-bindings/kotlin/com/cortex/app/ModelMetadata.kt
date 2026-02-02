package com.cortex.app

/**
 * GGUF metadata for a loaded model.
 * Contains key-value pairs from the model's metadata section.
 */
data class ModelMetadata(
    /** Map of all metadata key-value pairs */
    val metadata: Map<String, String> = emptyMap(),
    
    /** Model architecture (e.g., "llama", "qwen2", "phi3") */
    val architecture: String? = null,
    
    /** Quantization type (e.g., "Q4_K_M", "Q5_K_S") */
    val quantization: String? = null,
    
    /** Model file format version */
    val fileType: String? = null,
    
    /** Size of each tensor in the model (in bits) */
    val tensorType: String? = null,
    
    /** Number of attention heads */
    val attentionHeads: Int? = null,
    
    /** Model context length from training */
    val trainingContextLength: Int? = null,
    
    /** Vocabulary size from training */
    val trainingVocabSize: Int? = null,
    
    /** RoPE frequency base */
    val ropeFreqBase: Float? = null,
    
    /** RoPE scaling factor */
    val ropeScale: Float? = null,
    
    /** Whether the model uses GQA (Grouped Query Attention) */
    val usesGqa: Boolean? = null,
    
    /** Expert count for MoE models */
    val expertCount: Int? = null,
    
    /** Expert used count for MoE models */
    val expertUsedCount: Int? = null,
    
    /** Model license */
    val license: String? = null,
    
    /** Model author/organization */
    val author: String? = null,
    
    /** Model URL/repository */
    val url: String? = null,
    
    /** Model description */
    val description: String? = null,
    
    /** Tags associated with the model */
    val tags: List<String> = emptyList()
) {
    companion object {
        /** Empty metadata for when model is not loaded */
        val EMPTY = ModelMetadata()
        
        /** Common metadata keys in GGUF files */
        const val KEY_ARCHITECTURE = "general.architecture"
        const val KEY_QUANTIZATION = "general.file_type"
        const val KEY_FILE_TYPE = "general.file_type"
        const val KEY_NAME = "general.name"
        const val KEY_AUTHOR = "general.author"
        const val KEY_DESCRIPTION = "general.description"
        const val KEY_LICENSE = "general.license"
        const val KEY_URL = "general.url"
        const val KEY_TAGS = "general.tags"
        const val KEY_LANGUAGES = "general.languages"
        const val KEY_DATASET = "general.dataset"
        const val KEY_ATTENTION_HEADS = "*.attention.head_count"
        const val KEY_CONTEXT_LENGTH = "*.context_length"
        const val KEY_VOCAB_SIZE = "*.vocab_size"
        const val KEY_ROPE_FREQ_BASE = "*.rope.freq_base"
        const val KEY_ROPE_SCALE = "*.rope.scale"
        const val KEY_USES_GQA = "*.attention.use_gqa"
        const val KEY_EXPERT_COUNT = "*.expert_count"
        const val KEY_EXPERT_USED_COUNT = "*.expert_used_count"
        
        /**
         * Parse common metadata from a raw metadata map.
         */
        fun parse(metadata: Map<String, String>): ModelMetadata {
            return ModelMetadata(
                metadata = metadata,
                architecture = metadata["general.architecture"],
                quantization = metadata["general.file_type"],
                fileType = metadata["general.file_type"],
                description = metadata["general.description"],
                author = metadata["general.author"],
                license = metadata["general.license"],
                url = metadata["general.url"],
                tags = metadata["general.tags"]
                    ?.split(",")
                    ?.map { it.trim() }
                    ?: emptyList(),
                attentionHeads = metadata["*.attention.head_count"]?.toIntOrNull(),
                trainingContextLength = metadata["*.context_length"]?.toIntOrNull(),
                trainingVocabSize = metadata["*.vocab_size"]?.toIntOrNull(),
                ropeFreqBase = metadata["*.rope.freq_base"]?.toFloatOrNull(),
                ropeScale = metadata["*.rope.scale"]?.toFloatOrNull(),
                expertCount = metadata["*.expert_count"]?.toIntOrNull(),
                expertUsedCount = metadata["*.expert_used_count"]?.toIntOrNull()
            )
        }
    }
    
    /**
     * Get a metadata value by key.
     * Supports wildcard patterns like "*.context_length"
     */
    fun get(key: String): String? {
        // Direct lookup
        if (metadata.containsKey(key)) {
            return metadata[key]
        }
        
        // Try wildcard pattern matching
        if (key.startsWith("*.")) {
            val suffix = key.substring(2)
            val matchingKey = metadata.keys.find { it.endsWith(suffix) }
            if (matchingKey != null) {
                return metadata[matchingKey]
            }
        }
        
        return null
    }
    
    /**
     * Get all metadata keys.
     */
    fun keys(): Set<String> = metadata.keys
    
    /**
     * Check if a key exists.
     */
    fun contains(key: String): Boolean = get(key) != null
    
    /**
     * Get the number of metadata entries.
     */
    fun size(): Int = metadata.size
    
    /**
     * Check if metadata is empty.
     */
    fun isEmpty(): Boolean = metadata.isEmpty()
    
    /**
     * Get a formatted summary of the model.
     */
    fun getSummary(): String {
        return buildString {
            appendLine("Model Metadata:")
            appendLine("  Architecture: ${architecture ?: "Unknown"}")
            appendLine("  Name: ${metadata["general.name"] ?: "N/A"}")
            appendLine("  Quantization: ${quantization ?: "N/A"}")
            appendLine("  Total entries: ${metadata.size}")
        }
    }
}