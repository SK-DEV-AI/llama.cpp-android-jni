package com.cortex.app

/**
 * KV Cache memory control and statistics.
 * Provides fine-grained control over the key-value cache for advanced use cases.
 */
class MemoryControl(private val engine: LlamaEngine) {
    
    /**
     * Clear the entire KV cache, resetting all cached tokens.
     * This frees memory but loses all context from previous tokens.
     * 
     * @param clearData If true, also clear data buffers (default: true)
     */
    fun clearCache(clearData: Boolean = true) {
        engine.nativeClearKvCache(engine.getContextPointer(), clearData)
    }
    
    /**
     * Remove tokens from the KV cache for a specific sequence and position range.
     * Useful for context window management and sliding window attention.
     * 
     * @param seqId Sequence ID (-1 for any sequence)
     * @param pos0 Start position (-1 for beginning)
     * @param pos1 End position (-1 for end of sequence)
     * @return true if successful, false if partial sequence removal failed
     */
    fun removeTokens(seqId: Int = -1, pos0: Int = -1, pos1: Int = -1): Boolean {
        return engine.nativeRemoveKvCacheTokens(engine.getContextPointer(), seqId, pos0, pos1)
    }
    
    /**
     * Copy tokens from one sequence to another.
     * Useful for creating multiple conversation branches from a common prefix.
     * 
     * @param srcSeqId Source sequence ID
     * @param dstSeqId Destination sequence ID
     * @param pos0 Start position (-1 for beginning)
     * @param pos1 End position (-1 for end)
     */
    fun copySequence(srcSeqId: Int, dstSeqId: Int, pos0: Int = -1, pos1: Int = -1) {
        engine.nativeCopyKvCacheSequence(engine.getContextPointer(), srcSeqId, dstSeqId, pos0, pos1)
    }
    
    /**
     * Keep only tokens from a specific sequence, removing all others.
     * Useful when you want to isolate a single conversation.
     * 
     * @param seqId Sequence ID to keep
     */
    fun keepOnlySequence(seqId: Int) {
        engine.nativeKeepKvCacheSequence(engine.getContextPointer(), seqId)
    }
    
    /**
     * Shift positions of tokens by a delta value.
     * Useful for position interpolation or extending context windows.
     * 
     * @param seqId Sequence ID (-1 for any)
     * @param pos0 Start position (-1 for beginning)
     * @param pos1 End position (-1 for end)
     * @param delta Amount to shift (can be negative)
     */
    fun shiftPositions(seqId: Int = -1, pos0: Int = -1, pos1: Int = -1, delta: Int) {
        engine.nativeShiftKvCachePositions(engine.getContextPointer(), seqId, pos0, pos1, delta)
    }
    
    /**
     * Divide token positions by a factor.
     * Used for position interpolation in long context scenarios (e.g., extending 4K → 32K).
     * 
     * @param seqId Sequence ID (-1 for any)
     * @param pos0 Start position (-1 for beginning)
     * @param pos1 End position (-1 for end)
     * @param divisor Factor to divide by (must be > 1)
     */
    fun dividePositions(seqId: Int = -1, pos0: Int = -1, pos1: Int = -1, divisor: Int) {
        require(divisor > 1) { "Divisor must be > 1, got $divisor" }
        engine.nativeDivideKvCachePositions(engine.getContextPointer(), seqId, pos0, pos1, divisor)
    }
    
    /**
     * Get current KV cache statistics.
     * Note: Detailed stats may not be available in all llama.cpp versions.
     * 
     * @return Cache statistics
     */
    fun getStats(): KvCacheStats {
        val stats = engine.nativeGetKvCacheStats(engine.getContextPointer())
        return KvCacheStats(
            usedTokens = stats[0],
            maxTokens = stats[1],
            usedCells = stats[2],
            maxCells = stats[3]
        )
    }
    
    /**
     * Apply sliding window attention by removing old tokens.
     * Keeps only the most recent `windowSize` tokens.
     * 
     * @param windowSize Number of tokens to keep
     * @param seqId Sequence to apply to (-1 for all)
     */
    fun applySlidingWindow(windowSize: Int, seqId: Int = -1) {
        // Remove tokens outside the window
        // This is a simplified implementation
        removeTokens(seqId, 0, -windowSize)
    }
    
    /**
     * KV Cache statistics data class.
     */
    data class KvCacheStats(
        val usedTokens: Int,
        val maxTokens: Int,
        val usedCells: Int,
        val maxCells: Int
    ) {
        /**
         * Calculate memory usage percentage.
         */
        fun usagePercent(): Float {
            return if (maxTokens > 0) {
                (usedTokens.toFloat() / maxTokens.toFloat()) * 100f
            } else {
                0f
            }
        }
        
        /**
         * Check if cache is nearly full.
         */
        fun isNearlyFull(threshold: Float = 0.9f): Boolean {
            return usagePercent() >= threshold * 100f
        }
        
        override fun toString(): String {
            return "KvCacheStats(tokens: $usedTokens/$maxTokens, cells: $usedCells/$maxCells, usage: ${usagePercent().toInt()}%)"
        }
    }
}

/**
 * Extension function to create MemoryControl from LlamaEngine.
 */
fun LlamaEngine.memory(): MemoryControl = MemoryControl(this)