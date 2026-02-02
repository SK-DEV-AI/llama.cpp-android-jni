package com.cortex.app

/**
 * Model metadata and information.
 * Provides details about the loaded LLM.
 */
data class ModelInfo(
    /** Model description/name */
    val description: String,
    
    /** Total number of parameters (e.g., 7_000_000_000 for 7B model) */
    val parameterCount: Long,
    
    /** Model size in bytes */
    val sizeBytes: Long,
    
    /** Embedding dimension size */
    val embeddingSize: Int,
    
    /** Number of layers */
    val layerCount: Int,
    
    /** Number of attention heads */
    val headCount: Int,
    
    /** Number of KV heads (for GQA/MQA) */
    val headCountKV: Int,
    
    /** Training context size */
    val contextSize: Int,
    
    /** Vocabulary size (number of tokens) */
    val vocabSize: Int,
    
    /** Whether model has encoder (BERT-style) */
    val hasEncoder: Boolean,
    
    /** Whether model has decoder (GPT-style) */
    val hasDecoder: Boolean,
    
    /** Whether model is recurrent (RNN) */
    val isRecurrent: Boolean,
    
    /** Chat template string (if any) */
    val chatTemplate: String?
) {
    /** Formatted parameter count (e.g., "7B", "13B") */
    val formattedParams: String
        get() = when {
            parameterCount >= 1_000_000_000 -> "${parameterCount / 1_000_000_000}B"
            parameterCount >= 1_000_000 -> "${parameterCount / 1_000_000}M"
            else -> "$parameterCount"
        }
    
    /** Formatted size (e.g., "4.2 GB") */
    val formattedSize: String
        get() = when {
            sizeBytes >= 1_073_741_824 -> "%.1f GB".format(sizeBytes / 1_073_741_824.0)
            sizeBytes >= 1_048_576 -> "%.1f MB".format(sizeBytes / 1_048_576.0)
            else -> "%.1f KB".format(sizeBytes / 1024.0)
        }
    
    companion object {
        /** Empty model info (when model not loaded) */
        val EMPTY = ModelInfo(
            description = "",
            parameterCount = 0,
            sizeBytes = 0,
            embeddingSize = 0,
            layerCount = 0,
            headCount = 0,
            headCountKV = 0,
            contextSize = 0,
            vocabSize = 0,
            hasEncoder = false,
            hasDecoder = false,
            isRecurrent = false,
            chatTemplate = null
        )
    }
}
